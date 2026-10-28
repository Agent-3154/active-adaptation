"""Playback-only PhysX contact reports projected onto the Isaac RGB image.

No contact inference from SDF or net body force. Preserve every reported contact
and friction anchor at every physics substep; display the last substep per RGB.
"""
import gzip
import json
from pathlib import Path
import numpy as np


def enable_contact_reports(stage):
    from pxr import PhysxSchema, UsdPhysics
    paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
            paths.append(str(prim.GetPath()))
    return paths


class ContactVisualization:
    def __init__(self, env, output, force_scale=.01, max_arrow=.20):
        from pxr import UsdPhysics
        import omni.physics.tensors as tensors
        import omni.replicator.core as rep
        if env.num_envs != 1:
            raise ValueError('Contact video diagnostics currently require one environment')
        self.env, self.scale, self.cap = env, force_scale, max_arrow
        stage = env.sim.get_initial_stage()
        self.bodies = enable_contact_reports(stage)
        if not self.bodies:
            raise RuntimeError('No rigid bodies registered for contact reporting')
        self.camera = rep.AnnotatorRegistry.get_annotator('camera_params')
        self.camera.attach([env._render_product])
        self.substep = 0
        self.pending = []
        self.last = None
        self.count = self.contact_count = self.friction_count = 0
        self.peak_force = self.peak_friction = 0.
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = gzip.open(self.output/'contacts.jsonl.gz', 'wt')
        # Explicit all-to-all filters include static collision geometry (ground/table).
        static = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                parent = prim
                while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    parent = parent.GetParent()
                if not parent:
                    static.append(str(prim.GetPath()))
        self.filters = self.bodies + static
        self.tensor_sim = tensors.create_simulation_view('torch')
        self.tensor_sim.set_subspace_roots('/')
        self.contact_view = self.tensor_sim.create_rigid_contact_view(
            self.bodies, [self.filters]*len(self.bodies), max_contact_data_count=65536)
        self.bodies = list(self.contact_view.sensor_paths)
        assert self.contact_view.filter_count == len(self.filters)
        print('Contact tensors:', len(self.bodies), 'bodies;', len(self.filters), 'filters', flush=True)
        env._pre_step_callbacks.append(self._before_physics)
        env._post_step_callbacks.append(self._after_physics)

    def _before_physics(self, substep):
        if substep == 0:
            self.pending = []

    def _after_physics(self, substep):
        # Copy before friction query: the API reuses its count/offset buffers.
        dt = self.env.physics_dt
        normal = [x.cpu().numpy().copy() for x in self.contact_view.get_contact_data(dt)]
        friction = [x.cpu().numpy().copy() for x in self.contact_view.get_friction_data(dt)]
        forces, points, normals, distances, counts, offsets = normal
        ff, fp, fc, fo = friction
        if counts.sum() >= 65536 or fc.sum() >= 65536:
            raise RuntimeError('Contact diagnostic capacity exhausted')
        for i, j in np.argwhere((counts > 0) | (fc > 0)):
            # A-B and B-A are equal/opposite reports of the same contact.
            # Show/store one orientation, force acting on the first actor.
            if self.filters[j] in self.bodies and self.bodies[i] >= self.filters[j]:
                continue
            rows, anchors = [], []
            start, count = int(offsets[i,j]), int(counts[i,j])
            for k in range(start, start+count):
                force = normals[k]*float(forces[k,0])
                rows.append(dict(position=points[k].tolist(), normal=normals[k].tolist(),
                    force=force.tolist(), impulse=(force*dt).tolist(), separation=float(distances[k,0])))
            start, count = int(fo[i,j]), int(fc[i,j])
            for k in range(start, start+count):
                anchors.append(dict(position=fp[k].tolist(),force=ff[k].tolist(),impulse=(ff[k]*dt).tolist()))
            self.pending.append(dict(substep=substep, actors=[self.bodies[i],self.filters[j]],
                contacts=rows, friction=anchors))

    def capture(self, result):
        e = self.env
        self.last = dict(frame=self.count, episode_id=int(e.episode_id[0]),
            episode_step=int(e.episode_length_buf[0]), motion_frame=int(e.command_manager.motion_steps[0])-1,
            done=bool(result['done'][0]), physics_dt=e.physics_dt, reports=self.pending)
        self.last['voc_active'] = bool(e.command_manager.voc.enabled and
            (self.last['episode_step']-1)*e.step_dt < e.command_manager.voc.duration-1e-7)
        self.log.write(json.dumps(self.last)+'\n')
        for r in self.pending:
            self.contact_count += len(r['contacts'])
            self.friction_count += len(r['friction'])
            for c in r['contacts']:
                self.peak_force = max(self.peak_force, float(np.linalg.norm(c['force'])))
            for c in r['friction']:
                self.peak_friction = max(self.peak_friction, float(np.linalg.norm(c['force'])))
        self.count += 1

    def overlay(self, image):
        import cv2
        image = np.array(image, copy=True)
        if self.last is None:
            return image
        camera = self.camera.get_data()
        view = np.asarray(camera['cameraViewTransform']).reshape(4, 4)
        projection = np.asarray(camera['cameraProjection']).reshape(4, 4)
        if self.count == 1:
            (self.output/'contact-camera.json').write_text(json.dumps(
                {k:np.asarray(v).tolist() for k,v in camera.items()}, indent=2))
        height, width = image.shape[:2]
        def project(p):
            xyzw = np.r_[p, 1.] @ view @ projection
            if xyzw[3] <= 1e-8:
                return None
            xy = xyzw[:2]/xyzw[3]
            return (int((xy[0]+1)*width/2), int((1-xy[1])*height/2))
        contacts, friction = [], []
        for r in self.last['reports']:
            if r['substep'] == self.env.decimation-1:
                contacts.extend(r['contacts'])
                friction.extend(r['friction'])
        visible = 0
        for rows, color, point_color in [(contacts,(40,255,100),(255,220,30)),
                                          (friction,(30,210,255),(30,210,255))]:
            for c in rows:
                p, f = np.array(c['position']), np.array(c['force'])
                magnitude = float(np.linalg.norm(f))
                start = project(p)
                if start is None:
                    continue
                if 0 <= start[0] < width and 0 <= start[1] < height:
                    visible += 1
                length = min(magnitude*self.scale, self.cap)
                end = project(p+f/max(magnitude,1e-12)*length)
                if end is not None:
                    cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=.2)
                    if magnitude*self.scale > self.cap:
                        cv2.circle(image, end, 3, (255,60,60), -1)
                cv2.circle(image, start, 3, point_color, -1, cv2.LINE_AA)
        texts = [
            f"Normal contacts (yellow/green) {len(contacts)} | friction anchors (cyan) {len(friction)}",
            f"Force = impulse / {self.env.physics_dt:.6f}s | arrows: {self.scale*100:.1f}cm/N; cap {self.cap*100:.0f}cm (red tip)",
            f"t={self.last['frame']*self.env.step_dt:.2f}s | episode {self.last['episode_id']} age={self.last['episode_step']*self.env.step_dt:.2f}s | VOC={int(self.last['voc_active'])}",
            f"Last physics substep; all reported points incl. occluded | in view {visible}/{len(contacts)+len(friction)}",
        ]
        for i, text in enumerate(texts):
            cv2.putText(image,text,(12,24+24*i),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(image,text,(12,24+24*i),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        return image

    def close(self):
        self.log.close()
        self.camera.detach([self.env._render_product.path])
        summary = dict(frames=self.count, contact_records=self.contact_count,
            friction_records=self.friction_count, peak_contact_force_N=self.peak_force,
            peak_friction_force_N=self.peak_friction, reported_bodies=self.bodies, filter_paths=self.filters,
            display='All reported contacts + friction anchors in final physics substep; x-ray overlay',
            force_convention='PhysX tensor contact normal force on actor0 = impulse / physics_dt; friction anchors separately; reciprocal pairs deduplicated',
            arrow_m_per_N=self.scale, arrow_cap_m=self.cap,
            raw_records='contacts.jsonl.gz contains every reported physics substep')
        (self.output/'contact-summary.json').write_text(json.dumps(summary,indent=2))
        if self.contact_count == 0:
            raise RuntimeError('No PhysX contact reports received; contact visualization is invalid')

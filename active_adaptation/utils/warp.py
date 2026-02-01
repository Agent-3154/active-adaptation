import torch
import warp as wp
from jaxtyping import Float


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    # inputs
    mesh_id: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    min_dist: float,
    max_dist: float,
    # outputs
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distances: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ray_start = ray_starts[tid] + min_dist * ray_directions[tid]
    ray_dir = ray_directions[tid]
    result = wp.mesh_query_ray(
        mesh_id,
        ray_start,
        ray_dir,
        max_dist,
    )
    if result.result:
        t = result.t
        ray_hits[tid] = ray_start + t * ray_dir
        ray_distances[tid] = t - min_dist
    else:
        ray_distances[tid] = max_dist


@wp.kernel(enable_backward=False)
def raycast_mesh_grouped_kernel(
    # inputs
    mesh_id: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    # groups: wp.array(dtype=wp.int32),
    roots: wp.array(dtype=wp.int32),
    min_dist: float,
    max_dist: float,
    # outputs
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distances: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # mesh_root = wp.mesh_get_group_root(mesh_id, groups[tid])
    mesh_root = roots[tid]
    ray_start = ray_starts[tid] + min_dist * ray_directions[tid]
    ray_dir = ray_directions[tid]
    result = wp.mesh_query_ray(
        mesh_id,
        ray_start,
        ray_dir,
        max_dist,
        mesh_root,
    )
    if result.result:
        t = result.t
        ray_hits[tid] = ray_start + t * ray_dir
        ray_distances[tid] = t - min_dist
    else:
        ray_distances[tid] = max_dist


@wp.kernel(enable_backward=False)
def mesh_get_group_root_kernel(
    mesh_id: wp.uint64,
    groups: wp.array(dtype=wp.int32),
    # outputs
    roots: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    roots[tid] = wp.mesh_get_group_root(mesh_id, groups[tid])


def mesh_get_group_root(mesh: wp.Mesh, groups: torch.Tensor):
    shape = groups.shape
    groups = groups.reshape(-1)
    roots = torch.empty(len(groups), dtype=torch.int32, device=groups.device)
    wp.launch(
        kernel=mesh_get_group_root_kernel,
        dim=len(groups),
        inputs=[
            mesh.id,
            wp.from_torch(groups, dtype=wp.int32, return_ctype=True),
            wp.from_torch(roots, dtype=wp.int32, return_ctype=True),
        ],
    )
    return roots.reshape(shape)


def raycast_mesh(
    ray_starts: Float[torch.Tensor, "N 3"],
    ray_directions: Float[torch.Tensor, "N 3"],
    mesh: wp.Mesh,
    roots: Float[torch.Tensor, "N"] = None,
    min_dist: float = 0.01,
    max_dist: float = 100.0,
    sync: bool = False,
):
    shape = ray_starts.shape
    ray_starts = ray_starts.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    num_rays = ray_starts.shape[0]

    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3, return_ctype=True)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3, return_ctype=True)

    ray_hits = torch.empty(ray_starts.shape, device=ray_starts.device)
    ray_distances = torch.empty(ray_starts.shape[:-1], device=ray_starts.device)

    outputs = [
        wp.from_torch(ray_hits, dtype=wp.vec3, return_ctype=True),
        wp.from_torch(ray_distances, dtype=wp.float32, return_ctype=True),
    ]
    if roots is None:
        wp.launch(
            kernel=raycast_mesh_kernel,
            dim=ray_starts.shape[0],
            inputs=[
                mesh.id,
                ray_starts_wp,
                ray_directions_wp,
                min_dist,
                max_dist,
            ],
            outputs=outputs,
        )
    else:
        roots_wp = wp.from_torch(roots.reshape(num_rays), dtype=wp.int32, return_ctype=True)
        wp.launch(
            kernel=raycast_mesh_grouped_kernel,
            dim=ray_starts.shape[0],
            inputs=[
                mesh.id,
                ray_starts_wp,
                ray_directions_wp,
                roots_wp,
                min_dist,
                max_dist,
            ],
            outputs=outputs,
        )
    if sync:
        wp.synchronize()
    return ray_hits.reshape(shape), ray_distances.reshape(*shape[:-1], 1)

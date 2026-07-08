from pink import PinkNoiseProcess, cn
import numpy as np

class ParallelPinkNoiseProcess(PinkNoiseProcess):
    def __init__(self, size, scale=1, max_period=None, rng=None):
        super().__init__(size, scale, max_period, rng)


    def reset(self, mask: np.ndarray | None = None):
        """Reset the buffer with a new time series."""
        if not isinstance(mask, np.ndarray):
            self.buffer = cn.powerlaw_psd_gaussian(
                    exponent=self.beta, size=self.size, fmin=self.minimum_frequency, rng=self.rng)
            self.idx = np.zeros(shape=self.size[:1])
        else:
            self.buffer[mask] = cn.powerlaw_psd_gaussian(
                exponent=self.beta, size=[int(mask.sum()), ] + self.size[1:], fmin=self.minimum_frequency, rng=self.rng
            )
            self.idx[mask] = 0


    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            mask = self.idx >= self.time_steps
            if mask.any():
                self.reset(mask)

            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]

    def sample_one(self):
            """
            Sample 1 timestep from the colored noise process (Optimized for T=1 with vectorization).
            """
            mask = self.idx >= self.time_steps
            if mask.any():
                self.reset(mask)

            env_indices = np.arange(self.size[0])
            
            ret = self.buffer[env_indices, :, self.idx.astype(int)]

            self.idx += 1

            return self.scale * ret
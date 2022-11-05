# Module for custom ops, typically mpi4jax
import jax
import jax.numpy as jnp
import mpi4jax


def fft3d(arr, comms=None):
    """ Computes forward FFT, note that the output is transposed
    """
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    # First FFT along z
    arr = jnp.fft.fft(arr)  # [x, y, z]
    # Perform single gpu or distributed transpose
    if comms == None:
        arr = arr.transpose([1, 2, 0])
    else:
        arr = arr.reshape(shape[:-1]+[nx, shape[-1] // nx])
        #arr = arr.transpose([2, 1, 3, 0])  # [y, z, x]
        arr = jnp.einsum('ij,xyjz->iyzx', jnp.eye(nx), arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
        arr, token = mpi4jax.alltoall(arr, comm=comms[0])
        arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [y, z, x]

    # Second FFT along x
    arr = jnp.fft.fft(arr)
    # Perform single gpu or distributed transpose
    if comms == None:
        arr = arr.transpose([1, 2, 0])
    else:
        arr = arr.reshape(shape[:-1]+[ny, shape[-1] // ny])
        #arr = arr.transpose([2, 1, 3, 0])  # [z, x, y]
        arr = jnp.einsum('ij,yzjx->izxy', jnp.eye(ny), arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
        arr, token = mpi4jax.alltoall(arr, comm=comms[1], token=token)
        arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z, x, y]

    # Third FFT along y
    return jnp.fft.fft(arr)


def ifft3d(arr, comms=None):
    """ Let's assume that the data is distributed accross x
    """
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    # First FFT along y
    arr = jnp.fft.ifft(arr)  # Now [z, x, y]
    # Perform single gpu or distributed transpose
    if comms == None:
        arr = arr.transpose([0, 2, 1])
    else:
        arr = arr.reshape(shape[:-1]+[ny, shape[-1] // ny])
        # arr = arr.transpose([2, 0, 3, 1])  # Now [z, y, x]
        arr = jnp.einsum('ij,zxjy->izyx', jnp.eye(ny), arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
        arr, token = mpi4jax.alltoall(arr, comm=comms[1])
        arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z,y,x]

    # Second FFT along x
    arr = jnp.fft.ifft(arr)
    # Perform single gpu or distributed transpose
    if comms == None:
        arr = arr.transpose([2, 1, 0])
    else:
        arr = arr.reshape(shape[:-1]+[nx, shape[-1] // nx])
        # arr = arr.transpose([2, 3, 1, 0])  # now [x, y, z]
        arr = jnp.einsum('ij,zyjx->ixyz', jnp.eye(nx), arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
        arr, token = mpi4jax.alltoall(arr, comm=comms[0], token=token)
        arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [x,y,z]

    # Third FFT along z
    return jnp.fft.ifft(arr)


def halo_reduce(arr, halo_size, comms=None):
    if halo_size <= 0:
        return arr

    # Perform halo exchange along x
    rank_x = comms[0].Get_rank()
    size_x = comms[0].Get_size()
    margin = arr[-2*halo_size:]
    left, token = mpi4jax.sendrecv(margin, margin,
                                   (rank_x-1) % size_x,
                                   (rank_x+1) % size_x,
                                   comm=comms[0])
    margin = arr[:2*halo_size]
    right, token = mpi4jax.sendrecv(margin, margin,
                                    (rank_x+1) % size_x,
                                    (rank_x-1) % size_x,
                                    comm=comms[0], token=token)

    arr = arr.at[:2*halo_size].add(left)
    arr = arr.at[-2*halo_size:].add(right)

    # Perform halo exchange along y
    rank_y = comms[1].Get_rank()
    size_y = comms[1].Get_size()
    margin = arr[:, -2*halo_size:]
    left, token = mpi4jax.sendrecv(margin, margin,
                                   (rank_y-1) % size_y,
                                   (rank_y+1) % size_y,
                                   comm=comms[1], token=token)
    margin = arr[:, :2*halo_size]
    right, token = mpi4jax.sendrecv(margin, margin,
                                    (rank_y+1) % size_y,
                                    (rank_y-1) % size_y,
                                    comm=comms[1], token=token)
    arr = arr.at[:, :2*halo_size].add(left)
    arr = arr.at[:, -2*halo_size:].add(right)

    return arr


def meshgrid3d(shape, comms=None):
    if comms is not None:
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

        coords = [jnp.arange(shape[0]//nx),
                  jnp.arange(shape[1]//ny)] + [jnp.arange(s) for s in shape[2:]]
    else:
        coords = [jnp.arange(s) for s in shape[2:]]

    return jnp.stack(jnp.meshgrid(*coords), axis=-1).reshape([-1, 3])


def zeros(shape, comms=None):
    """ Initialize an array of given global shape
    partitionned if need be accross dimensions.
    """
    if comms is None:
        return jnp.zeros(shape)

    nx = comms[0].Get_size()
    ny = comms[1].Get_size()

    return jnp.zeros([shape[0]//nx, shape[1]//ny]+list(shape[2:]))


def normal(key, shape, comms=None):
    """ Generates a normal variable for the given
    global shape.
    """
    if comms is None:
        return jax.random.normal(key, shape)

    nx = comms[0].Get_size()
    ny = comms[1].Get_size()

    return jax.random.normal(key,
                             [shape[0]//nx, shape[1]//ny]+list(shape[2:]))

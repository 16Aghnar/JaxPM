import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import fftk, LDL_kernel, smoothing_kernel


def LDL_displacement_layer(pos, cosmo, params):
    """
    Computes the LDL displacement layer as defined in 2010.02926 equation (1) and (3)
    Parameters:
    -----------
    pos: array
      Array of particles positions
    cosmo:
      cosmology
    params: list of 5 floats
      LDL parameters for displacement
      alpha, kl, ks, n, gamma
    Returns:
    --------
    dpos_ldl: array
      displacement
    """
    #generate kvec
    kvec = fftk(mesh_shape)
    # turn DM particles into density map
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    # load parameters
    alpha, kl, ks, n, gamma = params
    # source term, simple power law
    f_delta = (1+delta)**gamma
    # in fourier space
    f_delta_k = jnp.fft.rfftn(f_delta)
    # Green operator
    LDL_range=LDL_kernel(kvec, kl, ks, n)
    pot_k_ldl=(f_delta_k)*LDL_range
    # gradient and displacement field
    forces_ldl= jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k_ldl), pos) 
                      for i in range(3)],axis=-1)
    # scale of displacement
    dpos_ldl = forces_ldl*alpha
    
    return dpos_ldl

def LDL_activation_layer(state, params):
    """
    Computes the LDL activation layer as defined in 2010.02926 equation (5)
    Parameters:
    -----------
    state: array
      Array of particles positions
    params: list of 3 floats
      LDL parameters for activation
      b1, b0, mu
    Returns:
    --------
    LDL_pred: array
      emulated baryonic field
    """
    # load params
    b1, b0, mu = params
    # from particules into density map
    delta_2 = cic_paint(jnp.zeros(mesh_shape), state)
    # return non linear activation of map
    return jax.nn.relu(b1*(1+delta_2)**mu - b0)

def LDL_prediction(pos, cosmo, params):
    """
    Computes the LDL displacement layer as defined in 2010.02926 equation (1) and (3)
    Parameters:
    -----------
    pos: array
      Array of particles positions
    cosmo:
      cosmology
    params: list of 13 floats
      LDL parameters for 2 displacements layer and 1 activation layer, 13 parameters
    Returns:
    --------
    LDL_pred: array
      displacement
    """
    # First displacement layer
    ldlized_state_1 = pos + LDL_displacement_layer(pos, cosmo, params[:5])
    # Second displacement layer
    ldlized_state_2 = ldlized_state_1 + LDL_displacement_layer(ldlized_state_1, cosmo, params[5:-3])
    # Relu layer
    LDL_pred = LDL_activation_layer(ldlized_state_2, params[-3:])
    return LDL_pred
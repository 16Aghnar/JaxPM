import numpy as np
import jax
import jax.numpy as jnp
#import jax_cosmo as jc
from jaxpm.painting import cic_paint, cic_read
from jaxpm.kernels import fftk, LDL_kernel, smoothing_kernel, gradient_kernel
import haiku as hk


def LDL_displacement_layer(pos, mesh_shape, params):
    """
    Computes the LDL displacement layer as defined in 2010.02926 equation (1) and (3)
    Parameters:
    -----------
    pos: array
      Array of particles positions
    mesh_shape: list
      shape of the mesh
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
    pot_k_ldl=(f_delta_k)*LDL_range  #do f_delta_k * (1+neuralspline apply...). Just 1 layer of displacement.
    # gradient and displacement field
    forces_ldl= jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k_ldl), pos) 
                      for i in range(3)],axis=-1)
    # scale of displacement
    dpos_ldl = forces_ldl*alpha
    
    return dpos_ldl

def LDL_activation_layer(state, mesh_shape, params):
    """
    Computes the LDL activation layer as defined in 2010.02926 equation (5)
    Parameters:
    -----------
    state: array
      Array of particles positions
    mesh_shape: list
      shape of the mesh
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
    #b1 = jax.nn.softplus(b1) #bad idea

    # from particules into density map
    delta_2 = cic_paint(jnp.zeros(mesh_shape), state)
    # return non linear activation of map
    return jax.nn.relu(b1*(1+delta_2)**mu - b0)

def LDL_prediction(pos, mesh_shape, params):
    """
    Computes the LDL displacement layer as defined in 2010.02926 equation (1) and (3)
    Parameters:
    -----------
    pos: array
      Array of particles positions
    mesh_shape: list
      shape of the mesh
    params: list of 13 floats
      LDL parameters for 2 displacements layer and 1 activation layer, 13 parameters
    Returns:
    --------
    LDL_pred: array
      emulated baryonic field
    """
    # First displacement layer
    ldlized_state_1 = pos + LDL_displacement_layer(pos, mesh_shape, params[:5])
    # Second displacement layer
    ldlized_state_2 = ldlized_state_1 + LDL_displacement_layer(ldlized_state_1, mesh_shape, params[5:-3])
    # Relu layer
    LDL_pred = LDL_activation_layer(ldlized_state_2, mesh_shape, params[-3:])
    return LDL_pred

def _deBoorVectorized(x, t, c, p):
    """
    Evaluates S(x).

    Args
    ----
    x: position
    t: array of knot positions, needs to be padded as described above
    c: array of control points
    p: degree of B-spline
    """
    k = jnp.digitize(x, t) -1
    
    d = [c[j + k - p] for j in range(0, p+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x - t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter_Activation(hk.Module):
  """A rotationally invariant filter parameterized by 
  a b-spline with parameters specified by a small NN."""

  def __init__(self, n_knots=8, latent_size_ns2f=16, latent_size_act=128, name=None):
    """
    n_knots: number of control points for the spline  
    """
    super().__init__(name=name)
    self.n_knots = n_knots
    self.latent_size_ns2f = latent_size_ns2f
    self.latent_size_act = latent_size_act

  def __call__(self, x, par):
    """ 
    x: array, scale, normalized to fftfreq default
    par: array, cosmo and physical parameters + redshift. shape (7,)
    """
    net1 = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(jnp.atleast_1d(par)))
    net1 = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(net1))
    w1 = hk.Linear(self.n_knots+1)(net1) 
    k1 = hk.Linear(self.n_knots-1)(net1)
    # make sure the knots sum to 1 and are in the interval 0,1
    k1 = jnp.concatenate([jnp.zeros((1,)),jnp.cumsum(jax.nn.softmax(k1))])
    w1 = jnp.concatenate([jnp.zeros((1,)),w1])
    # Augment with repeating points
    ak1 = jnp.concatenate([jnp.zeros((3,)), k1, jnp.ones((3,))])
    
    net2 = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(jnp.atleast_1d(par)))
    net2 = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(net2))
    w2 = hk.Linear(self.n_knots+1)(net2) 
    k2 = hk.Linear(self.n_knots-1)(net2)
    # make sure the knots sum to 1 and are in the interval 0,1
    k2 = jnp.concatenate([jnp.zeros((1,)),jnp.cumsum(jax.nn.softmax(k2))])
    w2 = jnp.concatenate([jnp.zeros((1,)),w2])
    # Augment with repeating points
    ak2 = jnp.concatenate([jnp.zeros((3,)), k2, jnp.ones((3,))])
    # 
    net3 = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(jnp.atleast_1d(par)))
    net3 = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(net3))
    actpars = hk.Linear(7)(net3)

    return _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak1, w1, 3), \
           _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak2, w2, 3), \
           actpars

class NeuralSplineFourierFilter_Activation_4l(hk.Module):
  """A rotationally invariant filter parameterized by 
  a b-spline with parameters specified by a small NN."""

  def __init__(self, n_knots=8, latent_size_ns2f=16, latent_size_act=128, name=None):
    """
    n_knots: number of control points for the spline  
    """
    super().__init__(name=name)
    self.n_knots = n_knots
    self.latent_size_ns2f = latent_size_ns2f
    self.latent_size_act = latent_size_act
  
  def _cosmo_to_knots(self, par):
    net = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(jnp.atleast_1d(par)))
    net = jax.nn.leaky_relu(hk.Linear(self.latent_size_ns2f)(net))
    w = hk.Linear(self.n_knots+1)(net) 
    k = hk.Linear(self.n_knots-1)(net)
    # make sure the knots sum to 1 and are in the interval 0,1
    k = jnp.concatenate([jnp.zeros((1,)),jnp.cumsum(jax.nn.softmax(k))])
    w = jnp.concatenate([jnp.zeros((1,)),w])
    # Augment with repeating points
    ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])
    activ = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(jnp.atleast_1d(par)))
    activ = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(activ))
    activ = hk.Linear(2)(activ)
    return ak, w, activ

  def __call__(self, x, par):
    """ 
    x: array, scale, normalized to fftfreq default
    par: array, cosmo and physical parameters + redshift. shape (7,)
    """
    # neural splines networks
    ak1, w1, activ1 = self._cosmo_to_knots(par)
    ak2, w2, activ2 = self._cosmo_to_knots(par)
    ak3, w3, activ3 = self._cosmo_to_knots(par)
    ak4, w4, activ4 = self._cosmo_to_knots(par)
    # activatioon network
    net3 = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(jnp.atleast_1d(par)))
    net3 = jax.nn.leaky_relu(hk.Linear(self.latent_size_act)(net3))
    actpars = hk.Linear(3)(net3)

    return _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak1, w1, 3), \
           _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak2, w2, 3), \
           _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak3, w3, 3), \
           _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak4, w4, 3), \
           activ1, activ2, activ3, activ4, \
           actpars

def NS2F_displacement(pos, mesh_shape, alpha, gamma, pot_res):
    """
    Computes the NS2F particle displacements
    Parameters:
    -----------
    pos: array
      Array of particles positions
    mesh_shape: list
      shape of the mesh
    alpha : float
      amplitude of displacement
    gamma : float
      power index on the field
    pot_res: 
      NS2F filter
    Returns:
    --------
    dpos: array
      displacement
    """
    #generate kvec
    kvec = fftk(mesh_shape)
    # turn DM particles into density map
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    # source term, simple power law
    f_delta = (1+delta)**gamma
    # in fourier space
    f_delta_k = jnp.fft.rfftn(f_delta)
    # apply correction filter
    pot_k = f_delta_k *(1. + pot_res)
    # gradient and displacement field
    forces = jnp.stack([cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i)*pot_k), pos) 
                        for i in range(3)],axis=-1)
    # scale of displacement
    dpos = forces*alpha
    return dpos

def NS2F_activated(pos, mesh_shape, pot_res1, pot_res2, actpars):
    """
    pos is dm particles positions
    pars is cosmo + physical parameters
    """
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    kvec = fftk(mesh_shape)
    # compute a conditioned filter and parameters
    kk = jnp.sqrt(sum((ki/np.pi)**2 for ki in fftk(mesh_shape)))
    
    gamma1, alpha1, gamma2, alpha2, b0, b1, mu = actpars
    
    # First displacement layer
    state_1 = pos + NS2F_displacement(pos, mesh_shape, alpha1, gamma1, pot_res1)
    # Second displacement layer
    state_2 = state_1 + NS2F_displacement(state_1, mesh_shape, alpha2, gamma2, pot_res2)
    
    delta_2 = cic_paint(jnp.zeros(mesh_shape), state_2)
    # return non linear activation of map
    return jax.nn.relu(b1*(1+delta_2)**mu - b0) #b1*(1+delta_2)**mu - b0 #j

def NS2F_activated_4l(pos, mesh_shape, pot_res1, pot_res2, pot_res3, pot_res4,
                      activ1, activ2, activ3, activ4, actpars):
    """
    pos is dm particles positions
    pars is cosmo + physical parameters
    """
    delta = cic_paint(jnp.zeros(mesh_shape), pos)
    kvec = fftk(mesh_shape)
    # compute a conditioned filter and parameters
    kk = jnp.sqrt(sum((ki/np.pi)**2 for ki in fftk(mesh_shape)))
    
    gamma1, alpha1 = activ1
    gamma2, alpha2 = activ2
    gamma3, alpha3 = activ3
    gamma4, alpha4 = activ4
    b0, b1, mu = actpars
    
    # First displacement layer
    state_1 = pos + NS2F_displacement(pos, mesh_shape, alpha1, gamma1, pot_res1)
    # Second displacement layer
    state_2 = state_1 + NS2F_displacement(state_1, mesh_shape, alpha2, gamma2, pot_res2)
    # Second displacement layer
    state_3 = state_2 + NS2F_displacement(state_2, mesh_shape, alpha3, gamma3, pot_res3)
    # Second displacement layer
    state_4 = state_3 + NS2F_displacement(state_3, mesh_shape, alpha4, gamma4, pot_res4)
    
    delta_2 = cic_paint(jnp.zeros(mesh_shape), state_4)
    # return non linear activation of map
    return jax.nn.relu(b1*(1+delta_2)**mu - b0) #b1*(1+delta_2)**mu - b0 #j

class Cosmo2LDL(hk.Module):
  """A small NN computing LDL parameters out of inpu cosmo"""

  def __init__(self, n_layers=2, latent_size_mlp=128, name=None):
    """
    n_knots: number of control points for the spline  
    """
    super().__init__(name=name)
    self.n_layers = n_layers
    self.latent_size_mlp = latent_size_mlp

  def __call__(self, par):
    """ 
    par: array, cosmo and physical parameters
    """
    net2 = jax.nn.leaky_relu(hk.Linear(self.latent_size_mlp)(jnp.atleast_1d(par)))
    net2 = jax.nn.leaky_relu(hk.Linear(self.latent_size_mlp)(net2))
    actpars = hk.Linear(13)(net2)

    return actpars

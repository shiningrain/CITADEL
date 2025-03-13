import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions 

class Test(tf.Module):
  def __init__(self):
    self.log_likes_list = None
    self.i = tf.constant(0)

  @tf.function
  def __call__(self, samples):

    @tf.function
    def rnd():
        return tfd.Normal(0,1).sample()+ tfd.Normal(3,1).sample()
      
    if self.log_likes_list is None:
        self.log_likes_list = tf.TensorArray(tf.float32, size=samples) 

    def cond(x,i):
        return tf.less(i, samples) 

    def body(x,i):
        #x=x.write(i,tfm.reduce_sum(tfd.Normal(rnd(), 1).log_prob(0.4)))
        # AttributeError: 'TensorArray' object has no attribute 'mark_used'
        x.write(i,tfm.reduce_sum(tfd.Normal(rnd(), 1).log_prob(0.4))).mark_used()
        return x, i+1 

    self.log_likes_list, i = tf.while_loop(cond, body, [self.log_likes_list, self.i])

    self.log_likes = self.log_likes_list.stack()

    self.log_like = tfm.reduce_mean(self.log_likes)

    loss = self.log_like

    return loss


T= Test()
t= T(5)

T.log_likes, T.log_like, 
# the code in below is not running 
T.log_likes_list.stack()


import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy

class LocalPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(LocalPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            #extracted_features - unpack obsercation
            gps, rgb = tf.split(self.processed_obs, [2, -1], 1)
            rgb = tf.reshape(rgb, [-1]+kwargs['rgb_shape'])

            #three convolutional layer
            n_filters = [3,4,4,1]
            for i in range(3):
                rgb = tf.nn.conv2d(rgb,
                            filters = tf.Variable(tf.random_normal([2, 2, n_filters[i], n_filters[i+1]])),
                            strides = [1, 2, 2, 1],
                            padding = 'SAME',
                            name='pi_fc' + str(i))

            rgb = tf.layers.flatten(rgb)
            tmp_vec = tf.concat([gps, rgb], 1)
            tmp_vec = activ(tf.layers.dense(tmp_vec, 50, name='pi_fc' + str(i)))

            value_fn = tf.layers.dense(tmp_vec, 1, name='vf')
            pi_latent = tf.layers.dense(tmp_vec, 1, name='pi')
            vf_latent = tmp_vec

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})



from stable_baselines3 import PPO

from PPONetwork import PPONetwork, PPOModel


class mainTichuSimulation(object):
    def __init__(self, sess, *, inpDim=412, nGames=8, nSteps=20, nMiniBatches=4, nOptEpochs=5, lam=0.95, gamma=0.995,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, minLearningRate=0.000001, learningRate, clipRange,
                 saveEvery=500):

        self.trainingNetwork = PPONetwork(sess, inpDim, 1695, "trainNet")
        self.trainingModel = PPOModel(sess, self.trainingNetwork, inpDim, 1695, ent_coef, vf_coef, max_grad_norm)
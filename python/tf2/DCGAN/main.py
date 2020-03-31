import tensorflow as tf
from model import DCGAN
from trainer import Trainer
from experiments import Experiments
from utils import get_config_from_json

def main():

    config = get_config_from_json('config.json')
    # create an instance of the model
    model = DCGAN(config)
    # create tensorboard logger
    logger = Logger(config, model)
    # create experiments instance
    experiments = Experiments(config, model)
    # create trainer instance
    trainer = Trainer(config, model, experiments)
    # train the model
    trainer.train()

if __name__ == '__main__':
    main()

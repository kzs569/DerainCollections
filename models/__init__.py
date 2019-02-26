from models.rescan import RESCAN
from models.pix2pix import Pix2Pix_Generator, Pix2Pix_Discriminator
from models.did_mdn import DenseDerain
from models.att_gan import Att_GAN_Discriminator, Att_GAN_Generator
from models.pan import PAN_Discriminator


def get_model(name):
    """get_model

    :param name:
    """
    return {
        "rescan": RESCAN,
        "pix2pix_g": Pix2Pix_Generator,
        "pix2pix_d": Pix2Pix_Discriminator,
        "did_mdn": DenseDerain,
        "attgan_g": Att_GAN_Generator,
        "attgan_d": Att_GAN_Discriminator,
        'pan_g': Pix2Pix_Generator,
        'pan_d': PAN_Discriminator,
    }[name]

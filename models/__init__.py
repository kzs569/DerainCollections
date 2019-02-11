from models.rescan import RESCAN
from models.pix2pix import Pix2Pix_Generator, Pix2Pix_Discriminator


def get_model(name):
    """get_model

    :param name:
    """
    return {
        "rescan": RESCAN,
        "pix2pix_g": Pix2Pix_Generator,
        "pix2pix_d": Pix2Pix_Discriminator,
    }[name]

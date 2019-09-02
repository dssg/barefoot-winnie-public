import os
import yaml
from barefoot_winnie.d00_utils.get_project_directory import get_project_directory
import logging
setup_dir = get_project_directory()

PARAMETERS_FILE = os.path.join(setup_dir, 'conf/base/parameters.yml')
CATALOG_FILE = os.path.join(setup_dir, 'conf/base/catalog.yml')


def get_parameter_yaml(parameter_key):
    """returns a specific parameter saved in the yml file"""
    with open(PARAMETERS_FILE) as stream:
        parameters = None
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

        return parameters[parameter_key]


def get_path_catalog(data_set_nickname):
    """ Returns a path of a specific item in the catalog file """
    with open(CATALOG_FILE) as stream:
        parameters = None
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

        return parameters[data_set_nickname]['filepath']

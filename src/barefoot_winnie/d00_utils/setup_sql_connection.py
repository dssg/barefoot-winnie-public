from sqlalchemy import create_engine
import yaml
import os
from barefoot_winnie.d00_utils.get_project_directory import get_project_directory


def setup_sql_connection():
    setup_dir = get_project_directory()
    credentials_path = os.path.join(setup_dir, 'conf', 'local', 'credentials.yml')
    with open(credentials_path, 'r') as credential_file:
        credentials = yaml.safe_load(credential_file)

    parameter_path = os.path.join(setup_dir, 'conf', 'base', 'parameters.yml')
    with open(parameter_path, 'r') as parameter_file:
        parameters = yaml.safe_load(parameter_file)

    db_host = parameters['DATABASE_PARAMS']['db_host']
    db_name = parameters['DATABASE_PARAMS']['db_name']
    db_user = credentials['dssg']['username']
    db_pass = credentials['dssg']['password']

    conn = create_engine('mysql+pymysql://%s:%s@%s/%s' %
                         (db_user, db_pass, db_host, db_name),
                         encoding='latin1')
    return conn

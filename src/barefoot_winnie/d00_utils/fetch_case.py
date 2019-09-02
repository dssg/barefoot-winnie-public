import pandas as pd
from sqlalchemy import select, MetaData, Table
from barefoot_winnie.d00_utils.setup_sql_connection import setup_sql_connection


def fetch_case(case_id=1):
    """ For a given case ID, gets the case information from SQL
    :param case_id: Case ID to be retrieved from the database
    :return: Case information as a pandas dataframe
    """

    conn = setup_sql_connection()

    metadata = MetaData(bind=None)
    table = Table('cases', metadata, autoload=True, autoload_with=conn)
    sql_query = select([table]).where(table.c.id == case_id)
    case_data = list(conn.execute(sql_query).fetchall())
    case_df = pd.DataFrame(case_data, columns=list(case_data[0].keys()))

    conn.dispose()

    return case_df

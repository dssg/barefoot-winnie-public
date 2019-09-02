import sqlalchemy
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import text


def create_bios_tables(conn):
    """ Creates the specified table in the BIOS MySQL database
    :param conn: A SQL connection
    :return: None
    """
    meta = MetaData()

    # The table definition
    recommended_responses_tbl = Table(
        'recommended_responses_3', meta,
        Column('id', sqlalchemy.types.INTEGER(), autoincrement=True, primary_key=True, nullable=False),
        Column('case_id', sqlalchemy.types.INTEGER(), nullable=False),
        Column('recommended_response', sqlalchemy.types.Text(collation='utf8mb4_unicode_ci'), nullable=False),
        Column('response_rank', sqlalchemy.types.INTEGER(), nullable=False)
    )

    meta.create_all(conn)
    conn.execute(text('ALTER TABLE recommended_responses MODIFY case_id Integer(10)'))

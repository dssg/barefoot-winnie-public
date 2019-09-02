from barefoot_winnie.d05_reporting.create_bios_tables import create_bios_tables
from barefoot_winnie.d00_utils.setup_sql_connection import setup_sql_connection

conn = setup_sql_connection()
create_bios_tables(conn)

import psycopg2

dbname = 'sketches'
database=dbname
port = 5432
host = 'localhost'
#password = 'INSERT PASSWORD'
#user='INSERT USER HERE'

class Database(object):
    def __init__(self):
        self.conn = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()
    def execute(self, **kwargs):
        self.cur.execute(**kwargs)
    def rollback(self):
        self.conn.rollback()
    def commit(self):
        self.conn.commit()

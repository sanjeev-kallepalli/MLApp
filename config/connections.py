import pyodbc

def get_mssql_connection():
    try:
        conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                                "Server=(localdb)\MSSQLLocalDB;"
                                "Database=master;")
        return conn
    except Exception as e:
        print(f"Error while connecting to mssql due to {e}")
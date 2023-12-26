import mysql.connector as db
from mysql.connector.connection_cext import CMySQLConnection


def insert_into_users(data: tuple, my_db: CMySQLConnection) -> None:

    '''
        Inserts Data into USERS Table

        Data: Data Containing all the Credentials of the user,
        my_db: MySQL Connector object
    '''

    db = my_db.cursor()

    query = '''
            INSERT INTO USERS (
            Username,
            Password,
            Email,
            Phone_Number, 
            Address
            )
            VALUES
            (
            %s,
            %s,
            %s, 
            %s,
            %s
            )
            '''

    db.execute(query, data)

    my_db.commit()

def validate_user(credentials: tuple,my_db: CMySQLConnection) -> bool:
    '''
        Returns True if the user is in the database and false if the user is not
        cretendials: A tuple containing the users name and password,
        my_db: A MySQL Connector object
    '''    
    db = my_db.cursor()

    query = '''
            SELECT user_id FROM Users WHERE username = %s AND password = %s
            ''' 

    db.execute(query, credentials)

    data = db.fetchall()

    if len(data):
        return data[0][0]
    
    return len(data)

def insert_into_post(USER_ID, MESSAGE, PREDICTION, IMAGEPATH, my_db : CMySQLConnection):
    credentials = (USER_ID, MESSAGE, PREDICTION, IMAGEPATH)

    query = '''
            INSERT INTO POSTS (
            USER_ID,
            MESSAGE,
            PREDICTION,
            IMAGEPATH
            ) VALUES (
            %s,
            %s,
            %s,
            %s
            )
        '''
    db = my_db.cursor()

    db.execute(query, credentials)

    my_db.commit()

def get_post_data(id, my_db:CMySQLConnection):

    db = my_db.cursor()

    query = '''
            SELECT * FROM POSTS WHERE user_id = %s ORDER BY post_id DESC;
    ''' 

    db.execute(query, (id,))

    data = db.fetchall()

    return data


if __name__ == "__main__":
    my_db = db.connect(
            host="localhost",
            user="root",
            password="root",
            database="websitedata"
        )

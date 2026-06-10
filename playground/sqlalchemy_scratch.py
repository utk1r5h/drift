from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///mydatabase.db', echo = True)


conn = engine.connect()

conn.execute(text("CREATE TABLE IF NOT EXISTS people (name str, age int)"))

conn.commit()


from sqlalchemy.orm import Session

session = Session(engine)

session.execute(text('INSERT INTO people (name, age) VALUES ("utkarsh", 22);'))

session.commit()

# this is the most basic way to work with sqlalchemy 

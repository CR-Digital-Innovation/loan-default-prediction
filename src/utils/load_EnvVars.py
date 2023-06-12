# pylint: disable=W0105
"""A simple utility to load all the variables from .env file make them available as global variables"""

from dotenv import dotenv_values

# Load the varibles from .env file
""" Default usage of dotenv_values

    import dotenv
    dotenv.dotenv_values('.env')

    You can replace with custom env file and you can leave blank if your filename is .env. Default value for load_dotenv is '.env'
"""

# Loop through each key-value pair in .env file and create respective variables

for key, value in dotenv_values().items():
    # Create variables dynamically using globals()
    globals()[key] = value

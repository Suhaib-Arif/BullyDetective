import os

def save_image_as(name: str):
    UPLOADSFOLDERNAME = os.path.join('.','static',"uploads")

    NUMBEROFFILES = str(len(os.listdir(UPLOADSFOLDERNAME)))
    os.makedirs(os.path.join(UPLOADSFOLDERNAME, NUMBEROFFILES))

    FINALPATH = os.path.join("uploads", NUMBEROFFILES, name + '.png')

    return FINALPATH





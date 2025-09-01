import os
import unicodedata

#dataset root path
root = "../raw/turkish-food"

def edit_filename(name):
    """
         Converts Turkish characters (ç,ş,ö,etc.) and spaces in the file name to ASCII format and underscores(_)

    :param name: str, The file name to be processed.
    :return: str, The cleaned filename with Turkish characters replaced and spaces converted to underscores.
    """

    #unicode normalization
    form_nfkd = unicodedata.normalize('NFKD', name)

    #Ignoring accent and special signs
    asci_ver = form_nfkd.encode('ASCII','ignore').decode('ASCII')

    #spaces -> (_)
    last_ver = asci_ver.replace(" ", "_")

    return last_ver

#Traverse files in folders
for dirpath, dirnames, filenames in os.walk(root):
    for filename in filenames:
        if filename.lower().endswith(".jpg"): #just (.jpg) file
            old_path = os.path.join(dirpath,filename)
            new_filename = edit_filename(filename)
            new_path = os.path.join(dirpath,new_filename)
            if old_path != new_path: # if the file name has not changed
                os.rename(old_path,new_path) # new path name
                print(f"Renamed: {old_path} -> {new_path}")

from ftplib import FTP
import tidyms as ms
import os

# this code downloads an example file from Metabolights via ftp
study_path = "pub/databases/metabolights/studies/public/MTBLS1919"
sample_path = os.path.join(study_path, "Applications/Centroid_data")
filename = "NZ_20200227_041.mzML"
ftp = FTP("ftp.ebi.ac.uk")
ftp.login()
ftp.cwd(sample_path)
with open(filename, "wb") as fin:
    ftp.retrbinary("RETR " + filename, fin.write)
ftp.close()

# specifying instrument and separation used in the experiments provides better
# default values for several functions used in
ms_data = ms.MSData.create_MSData_instance(
    filename, 
    ms_mode="centroid", 
    instrument="qtof",
    separation="uplc"
    )
roi_list = ms_data.make_roi()

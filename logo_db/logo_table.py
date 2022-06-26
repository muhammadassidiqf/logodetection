from flask_table import Table, Col, LinkCol

class Hasil(Table):
    logo_id = Col('Id', show=False)
    logo_nama = Col('Name')
    logo_filename = Col('Email')
    logo_upload = Col('Email')
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Credentials('databaseServices.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'url'
})

ref = db.refrence('Students')

data = {
    "id-1" : {
        "s_name" : ""
        "scop" : "",
        "father_name" : ""
    },
    "id-2" : {
        "s_name" : ""
        "scop" : "",
        "father_name" : ""
    }
}

for key,value in data.items():
    ref.child(key).set(value)
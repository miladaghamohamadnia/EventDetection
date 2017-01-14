from firebase import firebase
firebase = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
firebase.delete('001', None)
firebase.delete('002', None)
firebase.delete('003', None)

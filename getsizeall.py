from firebase import firebase
import time
fire_base = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)

while(1):
    result1 = fire_base.get('001', None)
    result2 = fire_base.get('002', None)
    result3 = fire_base.get('003', None)
    print '001' + ' ---  ', len( result1.items() )
    print '002' + ' ---  ', len( result2.items() )
    print '003' + ' ---  ', len( result3.items() )
    print ' -------------------'
    time.sleep(1)

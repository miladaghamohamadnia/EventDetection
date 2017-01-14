var Firebase = require("firebase");
// var google = require('googleapis');
// var storage = google.storage('v1');

var ref1 = new Firebase("https://esp001-864dd.firebaseio.com/001");
var ref2 = new Firebase("https://esp001-864dd.firebaseio.com/002");
var ref3 = new Firebase("https://esp001-864dd.firebaseio.com/003");

function loop(Ref){

        var k=0;
        Ref.once("value", function(snapshot) {


                var Total = snapshot.numChildren();
                // console.log(snapshot.exportVal());
                snapshot.forEach(function(childSnapshot) {
                        k=k+1;
                        // console.log(k)
                        var key = childSnapshot.key();
                        // childData will be the actual contents of the child
                        var childData = childSnapshot.val();
                        // console.log(key);
                        // console.log(childData);
                        if (k<Total-5000){
                            Ref.child(key).remove();
                        };
                        if (k==Total){
                            function f() {
                                console.log("inside")
                                process.exit()
                            }
                            setTimeout(f, 1*60*1000)
                        };
                });

                console.log(Total);
        });

};

loop(ref1)
loop(ref2)
loop(ref3)
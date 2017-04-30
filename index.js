var express = require('express');
var path = require('path');
var app = express();
app.use(express.static(path.join(__dirname, 'public')));

app.get('/radiation', function(req, res, next) {
    var spawn = require('child_process').spawn,
    py = spawn('python', ['AltitudeRad.py']),
    data = '',
    dataString = '';

    py.stdout.on('data', function(data) {
        console.log(data.toString());
        res.end(data.toString);
    });

    py.stdin.write(JSON.stringify(data));
    py.stdin.end();
});

app.listen(3000);

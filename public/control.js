var on = false;

function body_onkeydown(e){
    var key;
    if (window.event)
        key = window.event.keyCode;
    else if (e)
        key = e.which;

    switch (key)
    {
        case 27:
        on = !on;
        $('#scoreboard').css('visibility', 'visible');
        break;
    }
}

function getRadiation() {
    $.get({
        url: '/radiation'
    }).done(function(data) {
        console.log(data);
    }).fail(function(err) {
        console.log('Failed', err);
    });
}

$(document).ready(function () {
    getRadiation();
});
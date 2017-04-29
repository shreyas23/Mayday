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
        $('#scoreboard').css('visibility', on ? 'visible' : 'hidden');
        break;
    }
}

$(document).ready(function () {


});
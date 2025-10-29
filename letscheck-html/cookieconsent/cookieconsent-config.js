window.cookieconsent.initialise({
    palette:{
        popup: {background: "#edeff5"},
        button: {background: "#003062"},
    },
    
    content: {
        message: ' ',
        dismiss: 'Accept',
        link: 'Privacy Notice',
        href: 'https://www.nus.edu.sg/privacy-notice/'
    },
        cookie: {
        expiryDays:365,
        secure:true
    },
    position: 'bottom'
});

if (window.cookieconsent.hasInitialised){

    const cc_ContainerWidth = 1170;
    var cc_Window = document.getElementsByClassName('cc-window')[0];
    var cc_Button = document.getElementsByClassName('cc-btn')[0];
    var cc_chatboxExist = document.getElementById("watsonconv-floating-box");
    var cc_ScrollerExist = document.getElementById("scrollToTop");
    const cc_ChatboxWidth = 260;
    const cc_ScrollerWidth = 50;
    const cc_WindowPadding = '1em 16px 1em 16px';
    const cc_ButtonPadding = '0.3em 0.6em';
    const cc_MediaQuery = 'screen and (max-width: 414px) and (orientation: portrait), screen and (max-width: 736px) and (orientation: landscape)';
    //const cc_MediaQueryPortrait = 'screen and (orientation:portrait)';
    //const cc_MediaQueryLandscape = 'screen and (orientation:landscape)';

    const cc_MsgTitle = Object.assign(document.createElement('span'), {
        style: 'font-weight:700; font-size:14px; color:#333;',
        innerHTML: 'Privacy Notice'
    });

    const cc_Linebreak1 = document.createElement("br");

    const cc_MsgBody = document.createTextNode("This site uses cookies. By clicking accept or continuing to use this site, you agree to our use of cookies. For more details, please see our ");

    const cc_MsgLink = Object.assign(document.createElement('a'), {
        style: 'text-decoration:underline!important',
        href: 'https://www.nus.edu.sg/privacy-notice/',
        target: '_blank',
        innerHTML: 'Privacy Policy'
    });

    const cc_MsgBodyEnd = document.createTextNode('.');


    if (cc_Window.classList.contains('cc-floating')) {
        cc_Window.classList.add('cc-banner');
        cc_Window.classList.remove('cc-floating');
    }


    function cc_SetLayout(){
        let sidePad = 16;
        let newSidePad = window.innerWidth > cc_ContainerWidth + (sidePad * 2) ? (window.innerWidth - cc_ContainerWidth)/2 : sidePad;
        
        let matched = window.matchMedia(cc_MediaQuery).matches;
        if(matched){
            if (cc_chatboxExist){
                cc_Window.style.padding = '0.6em '+newSidePad+'px 5em '+newSidePad+'px';
            }else{
                cc_Window.style.padding = '0.6em '+newSidePad+'px 1em '+newSidePad+'px';
            }
            cc_Button.style.padding = '0.3em auto';
            cc_Button.style.width = '100%';
        }
        else{
            let buttonMarginTop = cc_Window.clientHeight > 110 ? 0 : 22;
            cc_ScrollerExist = document.getElementById("scrollToTop");
            cc_Button.style.padding = buttonMarginTop > 0 ? '0.3em 1.6em' : '0.3em 0.6em';
            cc_Button.style.marginTop = buttonMarginTop + 'px';


            if (cc_chatboxExist || cc_ScrollerExist){
                let extraPad = cc_chatboxExist ? cc_ChatboxWidth : cc_ScrollerWidth;
                if (window.innerWidth > cc_ContainerWidth + (extraPad * 2) ){
                    cc_Window.style.padding = '0.6em '+newSidePad+'px 2em '+newSidePad+'px';
                }else {
                    let newRightPad = newSidePad + extraPad;
                    cc_Window.style.padding = '1em '+newRightPad+'px 2em '+newSidePad+'px';
                }
            }else{
                cc_Window.style.padding = '0.6em '+newSidePad+'px 2em '+newSidePad+'px';
            }

        }
        
    }

    ['load', 'resize'].forEach(function(e) {
        window.addEventListener(e, cc_SetLayout);
    });


    document.getElementById('cookieconsent:desc').innerHTML = '';
    document.getElementById('cookieconsent:desc').style.fontSize = '14px';
    document.getElementById("cookieconsent:desc").appendChild(cc_MsgTitle);
    document.getElementById("cookieconsent:desc").appendChild(cc_Linebreak1);
    document.getElementById("cookieconsent:desc").appendChild(cc_MsgBody);
    document.getElementById("cookieconsent:desc").appendChild(cc_MsgLink);
    document.getElementById("cookieconsent:desc").appendChild(cc_MsgBodyEnd);

}
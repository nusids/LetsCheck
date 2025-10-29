$("#menu-header").load("menu/menu-header.html", function() {

    $("#nus-sidebar-off-canvas").ready(function() {
        buildSideBar("#nus-sidebar-off-canvas");

        $.getScript("js/offcanvas.js");
        $.getScript("js/common.js");
    })

});
$("#footer").load("menu/footer.html");
$(document).ready(function () {
	/* OFF CANVAS - SIDEBAR TOGGLE */
	$('[data-toggle="offcanvas"]').click(function () {
		$('.row-offcanvas').toggleClass('active');
	  });
	/* MENU DROPDOWN - NEXT LEVEL HOVER */
	$(".dropdown").hover(function() { 
		$(this).find('.dropdown-menu').first().stop(true, true).fadeIn("fast");
  }, function() { 
		$(this).find('.dropdown-menu').first().stop(true, true).fadeOut("fast");
  });
	/* WINDOW RESIZE FUNCTION  */
	$( window ).resize(function() {
		/* ABOVE 980px, Remove Off-Canvas Sidebar */
		if($( window ).width() >= 980) {
			$('body > div.nus-body-container').removeClass('active');
		}
	});
});
jQuery(document).ready(function ($) {
    if (/MSIE\s([\d.]+)/.test(navigator.userAgent) ? Number(RegExp.$1) < 10 : false) {
        $('html').addClass('old-ie');
    } else if (/constructor/i.test(window.HTMLElement)) {
        $('html').addClass('safari');
    }

    const $wrapper = $('body'),
        $inner = $('.nus-body-container'),
        $toggles = $('.off-canvas-toggle'),
        $offcanvas = $('.nus-sidebar-off-canvas'),
        $close = $('.nus-sidebar-off-canvas .close'),
        $btn = null,
        $nav = null,
        direction = 'left',
        $fixed = null;
    if (!$wrapper.length) return;
    $toggles.each(function () {
        const $this = $(this),
            $nav = $($this.data('nav')),
            effect = $this.data('effect'),
            direction = ($this.data('pos') === 'right') ? 'right' : 'left';
        $nav.addClass(effect).addClass('off-canvas-' + direction);
        const inside_effect = ['off-canvas-effect-3', 'off-canvas-effect-16', 'off-canvas-effect-7', 'off-canvas-effect-8', 'off-canvas-effect-14'];
        if ($.inArray(effect, inside_effect) === -1) {
            $inner.before($nav);
        } else {
            $inner.prepend($nav);
        }
    });

    $toggles.on('tap', function (e) {
        stopBubble(e);
        if ($wrapper.hasClass('off-canvas-open')) {
            oc_hide(e);
            return false;
        }

        const $btn = $(this);
        const $nav = $($btn.data('nav'));
        const $fixed = $inner.find('*').filter(function () {
            return $(this).css("position") === 'fixed';
        });

        $nav.addClass('off-canvas-current');

        const direction = ($btn.data('pos') == 'right') ? 'right' : 'left';

        $offcanvas.height($(window).height());

        const events = $(window).data('events');
        if (events && events.scroll && events.scroll.length) {
            const handlers = [];
            for (let i = 0; i < events.scroll.length; i++) {
                handlers[i] = events.scroll[i].handler;
            }
            $(window).data('scroll-events', handlers);
            $(window).off('scroll');
        }
        const scrollTop = ($('html').scrollTop()) ? $('html').scrollTop() : $('body').scrollTop(); // Works for Chrome, Firefox, IE...
        $('html').addClass('noscroll').css('top', -scrollTop).data('top', scrollTop);
        $('.t3-off-canvas').css('top', scrollTop);

        $fixed.each(function () {
            let $this = $(this),
                $parent = $this.parent(),
                mtop = 0;
            while (!$parent.is($inner) && $parent.css("position") === 'static') $parent = $parent.parent();
            mtop = -$parent.offset().top;
            $this.css({'position': 'absolute', 'margin-top': mtop});
        });

        $wrapper.scrollTop(scrollTop);
        $wrapper[0].className = $wrapper[0].className.replace(/\s*off\-canvas\-effect\-\d+\s*/g, ' ').trim() +
            ' ' + $btn.data('effect') + ' ' + 'off-canvas-' + direction;

        setTimeout(oc_show, 50);

        return false;
    });
    const oc_show = function () {
        $wrapper.addClass('off-canvas-open');
        $inner.on('click', oc_hide);
        $close.on('click', oc_hide);
        $offcanvas.on('click', stopBubble);
    };

    const oc_hide = function () {
        $inner.off('click', oc_hide);
        $close.off('click', oc_hide);
        $offcanvas.off('click', stopBubble);

        setTimeout(function () {
            $wrapper.removeClass('off-canvas-open');
        }, 100);

        setTimeout(function () {
            $wrapper.removeClass($btn.data('effect')).removeClass('off-canvas-' + direction);
            $wrapper.scrollTop(0);
            $('html').removeClass('noscroll').css('top', '');
            $('html,body').scrollTop($('html').data('top'));
            $nav.removeClass('off-canvas-current');
            $fixed.css({'position': '', 'margin-top': ''});
            if ($(window).data('scroll-events')) {
                const handlers = $(window).data('scroll-events');
                for (let i = 0; i < handlers.length; i++) {
                    $(window).on('scroll', handlers[i]);
                }
                $(window).data('scroll-events', null);
            }
        }, 550);

        if ($('html').hasClass('old-ie')) {
            const p1 = {}, p2 = {};
            p1['padding-' + direction] = 0;
            p2[direction] = -$('.t3-off-canvas').width();
            $inner.animate(p1);
            $nav.animate(p2);
        }
    };

    const stopBubble = function (e) {
        e.stopPropagation();
        return true;
    }
})

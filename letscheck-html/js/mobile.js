const buildSideBar = function (contElId) {
    const header =
        `
		<div class="header">
			<h2>Sidebar</h2>
			<button type="button" class="close" >×</button>
		</div>
		<div class="body">
			<div class="nus-module">
			<h3 class="title "><span>Navigation</span></h3>
		`;
    const footer = `</div></div>`;
    const navbar = $('#menu-header').find('.navbar ul.nav');

    const genLayer = function (ul, layerDepth) {
        const ulTagName = ul.prop('tagName');
        const els = ul.children();
        const navcls = layerDepth === 1 ? 'nav nav-pills nav-stacked' : (layerDepth > 1 ? "nav level" + (layerDepth) : null);
        const ul$ = $('<' + ulTagName + '>').addClass(navcls);
        els.each(function (i) {
            const cls = els[i].className;
            if (cls === 'dropdown-menu') {
                removeAllAttributes(els[i]);
                const subul$ = genLayer($(els[i]), layerDepth + 1);
                ul$.append(subul$);
            } else if ((cls === 'dropdown dropdown-submenu') || (cls === 'dropdown')) {
                removeAllAttributes(els[i]);
                const subul$ = genLayer($(els[i]), layerDepth);
                ul$.append(subul$);
            } else if (cls === 'dropdown-toggle') {
                filterHref(els[i]);
                const el$ = $(els[i]);
                el$.find('>b').remove();
                ul$.append(el$);
            } else {
                const el$ = $(els[i]);
                if ((layerDepth === 1) && (i === 0)) el$.addClass('current active');
                ul$.append(el$);
            }
        })
        return ul$
    }
    const ul$ = genLayer(navbar.clone(), 1)
    const html = header + ul$[0].outerHTML + footer;
    $(contElId).append(html);

    function filterHref(el) {
        const
            whitelist = ['href'],
            attributes = $.map(el.attributes, function (item) {
                return item.name;
            }),
            e = $(el)

        $.each(attributes, function (i, value) {
            if ($.inArray(value, whitelist) == -1) {
                e.removeAttr(value);
            }
        });
    }

    function removeAllAttributes(el) {
        const attributes = $.map(el.attributes, function (item) {
                return item.name;
            }),
            e = $(el)
        $.each(attributes, function (i, value) {
            e.removeAttr(value);
        })
    }
}

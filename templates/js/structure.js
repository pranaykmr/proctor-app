$(function(){
    
    // load dynamic navigation templates: 31Jul2017
    //$('.slideMenu .loginInfo').load('template/vitae-nav/vitae-nav__login-info.html');
    //$('.slideMenu .nav:first').load('template/vitae-nav/vitae-nav__citizen-nav.html');
    //$('.slideMenu .nav:last').load('template/vitae-nav/vitae-nav__app-nav.html');
    $( '<div id="navOverlay"></div>' ).insertAfter('.slideMenu');

    // load dynamic citizen context bar and modal: 31Jul2017
    //$('.citizenBar').load('template/citizen-context/vitae-layout__context-bar.html');
    //$('#citDetail .modal-content').load('template/citizen-context/vitae-modal__context-info.html');

    // load dynamic header with custom title: 31Jul2017
    /*function vitae__header(titleVal, hasBack){
        $('header').load('template/header/vitae-layout__header.html', function () {

            $('.headTitle > h3').html(titleVal);
            var backDisplay = hasBack ? 'inline-block' : 'none';
            $("header #back").css("display", backDisplay);
            if(!hasBack) $('.linkDvdr').hide();

        });
    }
    vitae__header('Add Professional Goal', true);*/


    // toggle slide navigation with body
    var $bodyEl = $('body'), $jsHideMenu = $('#tglNav');

    function hideSidedrawer() {
        $bodyEl.toggleClass('hideSlideMenu');
    }

    // toggle nav overlay for lower resolution
    $jsHideMenu.on('click', function(){
        hideSidedrawer();
        var sideBar = $('.slideMenu').css('left');
        if(sideBar == "-240px" ){
            $('#navOverlay').fadeIn();
            $(this).addClass('active');
        } else {
            $('#navOverlay').fadeOut();
            $(this).removeClass('active');
        }
    });



    // resize menu loading.......................
    $( window ).resize(function() {
        if($('body').hasClass('hideSlideMenu')){
            $('#navOverlay').fadeOut('slow');
        } else {
            $('#navOverlay').fadeIn('slow');
        }
    });
   

    $('#navOverlay').click(function () {
        $('body').toggleClass('hideSlideMenu');
        $jsHideMenu.removeClass('active');
        $(this).fadeOut();
    });




    // highlight respective label for input
    $("input, textarea, select").focus(function() {
        $("label[for='" + this.id + "']").addClass("labelFocus");
        }).blur(function() {
        $("label").removeClass("labelFocus");
    });

    // checking if citizen bar is available
    var $citizenDiv = $('.citizenBar');
    if (!($citizenDiv.length)){
        $('#container').addClass('noCitizenBar');
    }

    // checking if citizen bar is available
    var $footer = $('footer .footerBtn');
    if ($footer.length){
        $('#container').addClass('hasFooter');
    }

    // testing checkbox and applied seperate style
    /*var checkBx = $('.list .custInput_checkBox :checkbox');

    $(checkBx).change(function(){
        if($(this).is(':checked')){
            $(this).parent().parent().parent().addClass('checkedBox');
        } else {
            $(this).parent().parent().parent().removeClass('checkedBox');
        }
    });*/

    // toggle button script
    $('.tglBtn > a').click(function () {
        $(this).parent().find('a').removeClass('active');
        $(this).addClass('active');
    });

    // no network bar:
    var $noNetwork = $('.noNetwork'), $footerBtnBar = $('.footerBtn');
    if($noNetwork.length){
        $footer.addClass('network');
        $('#container').addClass('hasNoNetwork');
    }
    if($footerBtnBar.length){
        $footer.addClass('footerBtnBar');
    }

    // ADD SLIDEDOWN ANIMATION TO DROPDOWN //
    $('.dropdown').on('show.bs.dropdown', function (e) {
        $(this).find('.dropdown-menu').first().stop(true, true).slideDown(300);
        if($(this).find('.dropdown-menu > li:last-child').hasClass('syncModule')){
            $(this).find('.dropdown-menu > .syncModule').prev().find('a').css('border', 'none');
        }

    });
    $('.dropdown').on('hide.bs.dropdown', function (e) {
        $(this).find('.dropdown-menu').first().stop(true, true).slideUp(200);
    });




    // calling bootstrap popover with click: 17AUG 2017
    $('.info-popover').popover();

    // autosize textarea
    //autosize($('textarea'));
});



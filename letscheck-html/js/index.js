$('#twitter').click(function () {
    const t = this;
    onSearchButtonClick(t) && getTwatchApiTwitterResults();
})

$('#news').click(function () {
    const n = this;
    onSearchButtonClick(n) && getQuinApiResultForNewsArticles();
})

$('#science').click(function () {
    const s = this;
    onSearchButtonClick(s) && getQuinApiResultForScientificArticles();
})

$('#query-form').submit(function () {
    return submitQuery();
})

$('#query-submit-button').click(function () {
    submitQuery();
});

/** Displays the spinner while searching for Scientific Articles or News. */
function displayLoadingSpinner() {
    document.getElementById("loading-spinner").style.display = "inline-block";
}

/** Hides the spinner after searching for Scientific Articles or News. */
function hideLoadingSpinner() {
    document.getElementById("loading-spinner").style.display = "none";
}

/** Handles no results found when searching */

function showSearchErrorMessages() {
    const searchWord = document.getElementById("query-input").value;
    let searchWordDisplay = `"${searchWord}"`;
    if (searchWord === "") {
        searchWordDisplay = "empty";
    }
    document.getElementById("search-error-messages").innerHTML =
        `<br>
            <i class="fa fa-info-circle"></i>  
            Sorry, we couldn't find any results matching ${searchWordDisplay}. 
            <br>
            <br> 
            Remember to check your spelling and use the search term "COVID-19" for more accurate results. `
}

function removeFetchErrorMessages() {
    document.getElementById("search-error-messages").innerHTML = ``;
}

/** Removes current search data when initiating a new search */
function removeSearchData() {
    document.getElementById("search-results").style.display = "none";
    document.getElementById("search-results-content").innerHTML = ``;
    document.getElementById("search-results-content").style.display = "none";
    document.getElementById("search-results-credit").innerHTML = ``;
    document.getElementById("search-results-credit").style.display = "none";
}

/** Removes trend frame when initiating a new search */
function closeTrendFrame() {
    document.getElementById("trend-frame").src = ``;
    document.getElementById("trend-frame-container").style.display = "none";
}

/** Removes Twatch frame when initiating a new search */
function closeTwatchFrame() {
    document.getElementById("Twatch-frame").src = ``;
    document.getElementById("Twatch-frame-container").style.display = "none";
}

/** Shows the search results */
function activateSearchContent() {
    document.getElementById("search-results").style.display = "block";
    document.getElementById("search-count").style.display = "inline";
    document.getElementById("search-results-content").style.display = "block";
    document.getElementById("search-results-credit").style.display = "block";
    $('.result').show();
}

/** Removes previous Quinn results when initiating a new search */
function removePreviousQuinnResults() {
    closeTrendFrame();
    closeTwatchFrame();
    removeSearchData();
    removeFetchErrorMessages();
}

/** Removes previous results when initiating a new Twatch search */
function removePreviousResultsForTwatch() {
    hideLoadingSpinner();
    deactivateSearchContent();
    removeFetchErrorMessages();
}

/** Removes/hides the search results */
function deactivateSearchContent() {
    document.getElementById("search-results-content").style.display = "none";
    document.getElementById("search-results-credit").style.display = "none";
}

/** Executes the action  {@code promise} after {@code seconds} of timeout */
function timeOut(seconds, promise) {
    return new Promise(function (resolve, reject) {
        setTimeout(function () {
            reject(new Error("timeout"))
        }, seconds)
        promise.then(resolve, reject)
    })
}

/** After button is clicked, it's not allowed to be reclicked until the search results are out */
function onSearchButtonClick(button) {
    const allButtons = $('button.choices')
    allButtons.addClass('btn-default');
    allButtons.removeClass('btn-primary');
    $(button).addClass('btn-primary');
    $(button).removeClass('btn-default');
    $('#active-engine')[0].value = button.id;
    return checkQueryInput();
}

/** Submit the search query using the selected option */
function submitQuery() {
    const activeEngine = $('#active-engine')[0].value;
    if (activeEngine === 'twitter') {
        checkQueryInput() && getTwatchApiTwitterResults();
    } else if (activeEngine === 'news') {
        checkQueryInput() && getQuinApiResultForNewsArticles();
    } else if (activeEngine === 'science') {
        checkQueryInput() && getQuinApiResultForScientificArticles();
    }
    return false;
}

/** Fetches results from Quinn API for Scientific Articles and displays it */
function getQuinApiResultForScientificArticles() {
    removePreviousQuinnResults();
    displayLoadingSpinner();
    document.getElementById("search-error-messages").style.display = "block";
    const searchWord = parseQueryInput(document.getElementById("query-input").value);
    const apiUrl = `https://letscheck.nus.edu.sg/quin/api?source=cord&query=${searchWord}`;

    timeOut(FETCH_TIME_OUT_MILISECONDS, fetch(apiUrl)).then(response => {
        return response.json();
    }).then(response => {
        const numberOfResults = response.data.results.length;
        if (numberOfResults === 0) {
            throw Error;
        }

        const numSupporting = response.data.results.filter(r => r.nli_class === "entailment").length
        const numRefuting = response.data.results.filter(r => r.nli_class === "contradiction").length
        const veracityRating = response.data.veracity_rating;

        hideLoadingSpinner();
        if (veracityRating) {
            document.getElementById("search-results-content").innerHTML += displayVeracityRating(veracityRating);
            /* document.getElementById("search-results-content").innerHTML += resultTabs(numberOfResults, numSupporting, numRefuting); */
        }

        document.getElementById("search-results-content").innerHTML += `<div id="search-results-content-items"></div>`
        for (i = 0; i < numberOfResults; i++) {
            const currentSource = response.data.results[i];
            const title = currentSource.title.replace(/\\x../g, '').replace(/\\./g, '');
            const surl = currentSource.url.split(';')[0];
            const snippet = currentSource.snippet.replace(/\\x../g, '').replace(/\\./g, '');
            const authors = currentSource.authors;
            const journal = currentSource.journal;
            const nli_class = currentSource.nli_class;

            document.getElementById("search-results-content-items").innerHTML +=
                `<div class="result ${nli_class}" id="item${i}">
                   <div class="title">
                     <a href="${surl}" target = "-blank">${title}</a>
                   </div> 
                   <span class="text"> 
                     ...${snippet}...
                   </span><br><br> 
                   <div class="gray small">
                      ${authors}.
                      &nbsp;·&nbsp;${journal}
                   </div> 
                 </div>
                 <div class="line-break">
                   <br/><br/>
                 </div>    
                 `;
        }
        document.getElementById("search-results-credit").innerHTML +=
            `<small>
               The results are provided by <a href="https://quin.algoprog.com/">Quin</a>, a project by <a href="https://algoprog.com">Chris Samarinas</a>, <a href="https://www.comp.nus.edu.sg/~whsu/">Wynne Hsu</a>, <a href="https://www.comp.nus.edu.sg/~leeml/">Lee Mong Li</a>.
             </small>
             <br>`
        ;
        activateSearchContent();
    }).catch(err => {
        hideLoadingSpinner();
        showSearchErrorMessages();
        console.log(err);
    });
}

/** Fetches results from Quinn API for News and displays it. The method is almost the same
 but the apiUrl and response content */
function getQuinApiResultForNewsArticles() {
    removePreviousQuinnResults();
    displayLoadingSpinner();
    document.getElementById("search-error-messages").style.display = "block";

    const searchWord = parseQueryInput(document.getElementById("query-input").value);
    const apiUrl = `https://letscheck.nus.edu.sg/quin/api?source=news&query=${searchWord}`;

    timeOut(FETCH_TIME_OUT_MILISECONDS, fetch(apiUrl)).then(response => {
        return response.json();
    }).then(response => {
        const numberOfResults = response.data.results.length;
        if (numberOfResults === 0) {
            throw Error;
        }
        const numSupporting = response.data.results.filter(r => r.nli_class === "entailment").length
        const numRefuting = response.data.results.filter(r => r.nli_class === "contradiction").length
        const veracityRating = response.data.veracity_rating;
        hideLoadingSpinner();
        if (veracityRating) {
            document.getElementById("search-results-content").innerHTML += displayVeracityRating(veracityRating);
            /* document.getElementById("search-results-content").innerHTML += resultTabs(numberOfResults, numSupporting, numRefuting); */
        }

        document.getElementById("search-results-content").innerHTML += `<div id="search-results-content-items"></div>`
        for (i = 0; i < numberOfResults; i++) {
            const currentSource = response.data.results[i];
            const title = currentSource.title.replace(/\\x../g, '').replace(/\\./g, '');
            const url = currentSource.url;
            const snippet = currentSource.snippet.replace(/\\x../g, '').replace(/\\./g, '');
            const date = currentSource.date;
            const nli_class = currentSource.nli_class;

            document.getElementById("search-results-content-items").innerHTML +=
                `<div class="result ${nli_class}" id="item${i}">
                   <div class="title"> 
                     <a href=${url} target = "-blank">${title}</a>
                   </div> 
                   <span class="text"> 
                     ...${snippet}...
                   </span>
                   <div class="gray small">
                     ${date}
                   </div> 
                 </div>
                 <div class="line-break">
                   <br/><br/>
                 </div>    
                `;
        }
        document.getElementById("search-results-credit").innerHTML +=
            `<small>
               The results are provided by <a href="https://quin.algoprog.com/">Quin</a>, a project by <a href="https://algoprog.com">Chris Samarinas</a>, <a href="https://www.comp.nus.edu.sg/~whsu/">Wynne Hsu</a>, <a href="https://www.comp.nus.edu.sg/~leeml/">Lee Mong Li</a>.
             </small>
             <br>`
        ;
        activateSearchContent();
    }).catch(err => {
        hideLoadingSpinner();
        showSearchErrorMessages();
        console.log(err);
    });
}

/** Fetches and displays search results in Twatch frame */
function getTwatchApiTwitterResults() {
    closeTrendFrame();
    removePreviousResultsForTwatch();
    document.getElementById("search-error-messages").style.display = "none";
    const searchWord = parseQueryInput(document.getElementById("query-input").value);
    document.getElementById("Twatch-frame").src = `https://letscheck.nus.edu.sg/twatch/search?q=${searchWord}`;
    setIframeHeight('Twatch-frame', 1800);

    let TIME_OUT_TO_SHOW_FRAME = 500;
    setTimeout(function () {
        document.getElementById("Twatch-frame-container").style.display = "block";
    }, TIME_OUT_TO_SHOW_FRAME);
}

function checkQueryInput() {
    const f = document.getElementById("query-input");
    const m = document.getElementById("query-input-empty-message");
    let query = f.value.toLocaleLowerCase();
    const filteredList = ['covid', 'coronavirus', 'corona', 'virus']
    filteredList.forEach(w => {
        query = query.replaceAll(w, '')
    })

    query = query.replace(/\s\s+/g, ' ');
    query = query.replace(/[^a-zA-Z]/g, ' ');
    query = query.trim();

    if (!f.checkValidity()) {
        f.focus();
        m.textContent = 'Please enter your search query here.';
        return false;
    }

    if (query === '') {
        f.focus();
        m.textContent = 'Please type in a claim or a sentence.';
        return false;
    }

    m.textContent = "";
    return true;
}

function setIframeHeight(id, height) {
    const ifrm = document.getElementById(id);
    ifrm.style.height = "10px";
    ifrm.style.height = height + 4 + "px";
}
document.addEventListener('DOMContentLoaded', function () {
    $('#search-box-form').submit(function () {
        getQuinApiResult();
        return false;
    });
    fetch("https://letscheck.nus.edu.sg/twatchapi/api/custom", {
        method: 'post',
        body: '{"type": "trending_news"}',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    }).then((response) => {
        response.body.getReader().read().then((res) => {
            const re = new TextDecoder().decode(res.value);
            JSON.parse(re).forEach((news, i) => {
                document.querySelector(`#example-qn-${i}`).innerHTML = news;
            })
        });
    }).catch((error) => {
        console.log(error)
    });
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
    document.getElementById("search-error-messages").style.display = 'block';
    document.getElementById("search-error-messages").innerHTML =
        `<br>
            <i class="fa fa-info-circle"></i>  Sorry, we couldn't find any results matching ${searchWordDisplay}. <br>`;
}

function removeFetchErrorMessages() {
    document.getElementById("search-error-messages").innerHTML = ``;
    document.getElementById("search-error-messages").style.display = 'none';
}

/** Removes current search data when initiating a new search */
function removeSearchData() {
    document.getElementById("search-results").style.display = "none";
    document.getElementById("search-results-content").innerHTML = ``;
    document.getElementById("search-results-content").style.display = "none";
    document.getElementById("search-results-credit").style.display = "none";
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
    removeSearchData();
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

/** Fetches results from Quinn API for Scientific Articles and displays it */
function getQuinApiResult() {
    document.getElementById("examples-div").classList.add('hide');
    removePreviousQuinnResults();
    displayLoadingSpinner();

    const searchWord = parseQueryInput(document.getElementById("query-input").value);
    const apiUrl = `https://letscheck.nus.edu.sg/quin/api2?query=${searchWord}`;

    timeOut(FETCH_TIME_OUT_MILISECONDS, fetch(apiUrl)).then(response => {
        return response.json();
    }).then(response => {
        const numberOfResults = response.data.results.length;
        if (numberOfResults === 0) {
            throw Error;
        }

        const veracityRating = response.data.veracity_rating;

        hideLoadingSpinner();
        if (veracityRating) {
            document.getElementById("search-results-content").innerHTML += displayVeracityRating(veracityRating);
        }

        document.getElementById("search-results-content").innerHTML += `<div id="search-results-content-items"></div>`
        for (i = 0; i < numberOfResults; i++) {
            const currentSource = response.data.results[i];
            const title = currentSource.title.replace(/\\x../g, '').replace(/\\./g, '');
            const url = currentSource.url;
            const snippet = currentSource.snippet.replace(/\\x../g, '').replace(/\\./g, '');
            const nli_class = currentSource.nli_class;

            document.getElementById("search-results-content-items").innerHTML +=
                `<div class="result ${nli_class}" id="item${i}">
                        <div class="title"> 
                            <a href=${url} target="-blank">${title}</a>
                        </div>
                        <span class="text"> 
                            ...${snippet}...
                        </span>
                      </div>
                      <div class="line-break">
                        <br/>
                        <br/>
                      </div>
                    `;
        }
        activateSearchContent();
    }).catch(err => {
        hideLoadingSpinner();
        showSearchErrorMessages();
        console.log(err);
    });
}

document.querySelectorAll('.example-qn').forEach((item) => {
    item.addEventListener('click', (event) => {
        const query = event.target.innerHTML.trim();
        document.getElementById("query-input").value = query;
        event.target.blur();
    });
});

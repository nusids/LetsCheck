/** Colors of veracity ratings */
const veracityClasses = {
    "Probably True": "true",
    "Probably False": "false",
    "? Ambiguous": "ambiguous",
    "Not enough evidence": "unknown",
}

const veracityBefore = {
    "Probably True": "✔ ",
    "Probably False": "✘ ",
    "? Ambiguous": "",
    "Not enough evidence": "",
}

FETCH_TIME_OUT_MILISECONDS = 30000;

const showTab = (clicked, showClassName) => {
    console.log(clicked);
    $('.rtype').removeClass('selected');
    $(clicked).addClass('selected');
    $('.result').hide();
    $(showClassName).show();
}

/*
const resultTabs = (numAll, numSupporting, numRefuting) => (
    `
         <div id="rtypes" class="centered" style="display: block;">
           <span class="rtype selected" id="r_all" onclick="showTab(this, '.result')">all (${numAll})</span>
           <span class="rtype" id="r_supporting" onclick="showTab(this, '.entailment')">✔ supporting (${numSupporting})</span>
           <span class="rtype" id="r_refuting" onclick="showTab(this, '.contradiction')">✘ refuting (${numRefuting})</span>
         </div>
        `);
*/

const displayVeracityRating = (veracityRating) => (
    `
        <div class="veracity-rating">
          Veracity rating:
            <span class="${veracityClasses[veracityRating]}">
              ${veracityBefore[veracityRating] + veracityRating}
            </span>
        </div>
        <p class="disclaimer">
          Disclaimer: The veracity rating is still experimental and it is based on evidence from articles.<br/>
          For more reliable fact-checks you can refer to <a href="https://en.wikipedia.org/wiki/List_of_fact-checking_websites">fact-checking organizations</a>.
        </p>
    `
);

/** Cleans the search input query */
function parseQueryInput(sentence) {
    const words = sentence.split(" ").filter(word => word.length > 0);
    const spaceCharacter = "%20";
    return words.join(spaceCharacter);
}
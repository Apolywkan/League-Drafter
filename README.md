# League-Drafter

This program uses a web scraper to grab relevant winrate and matchup info for each provided champion in the draft. 
Once all the draft information is provided by the user, the web scraper grabs info from u.gg which utilizes the RIOT API
to calculate winrate and matchup statistics. 
By comparing the winrate and matchup statistics for all 5 lane matchups in the draft, a random forest algorithm calculates
the likelihood of you winning the game.

## Usage

The script will prompt you to enter the names of the given champions for each role. Enter the champion names in all lowercase, with no spaces. Omit special characters when the champion name requires them. Hit enter once the champion named is typed properly. Once all champions are entered wait for the algorithm to predict the outcome of the match

Examples:
"Dr. Mundo" -> drmundo
"Caitlyn" -> caitlyn
"Miss Fortune" -> missfortune

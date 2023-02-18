# Globo Recommendation Dataset

Sample data from July-September/2022 from a specific Feed on g1's news portal. A small sample is available within this directory. As an attempt to make it easier to use in news recommender research work, we tried to make the schema as similar as possible to Microsoft's MIND dataset.  

In Globo's recommender, every impression in trigged by a "recommendation strategy", which combines multiple algorithms. All recommendation impressions are subjected to an AB testing experiment, which chooses one specific "recommendation strategy" (AB testing alternative). 

Despite combining multiple algorithms in the same impression, there is always a main algorithm in every impression and for every user there is only one impression (1:1)


## Dataset Schema
* 
* news: metadata (150k articles),
* behaviors: impressions and user history (10M impressions; 4M users).

### Behaviors Dataset

| Field               | Type   | Example                              | Description        |
|---------------------|--------|--------------------------------------|--------------------|
| impressionId        | String | 7c5c78bd-3950-466b-afdc-138d5aa1fac8 | - | 
| impressionTimestamp | Long   | 1656714139838 | - |
| userId              | String | 17028942367096274688 | - |
| userType            | String | Non-Logged | Binary (Looged or Non-Logged) |
| clickedItem         | String | 9223031724623619632 | Id of the article clicked by the user on the news feed |
| clickedFeedPosition | Integer | 3 | Feed Position (from 1 to 20) clicked by the user |
| impressions         | Array<String-Integer> | 9223031724623613709-1,9223031724623615754-0, ... | List of article ids and if it was clicked or not by the user |
| historySize         | Integer | 8 | Number of articles clicked the user within 2 months |
| history             | Array<String> | 9223031724623608124,9223031724623612445, ... | Last articles clicked by the user in the last two months (limited to 50) |


![](../../../../../../../../../var/folders/1q/dzgrtt9s5478fmtf81cn31ch0000gn/T/TemporaryItems/NSIRD_screencaptureui_ut52eu/Screen Shot 2023-02-18 at 15.13.23.png)

### Articles Dataset

| Field           | Type    | Example                       | Description        |
|-----------------|---------|-------------------------------|--------------------|
| url             | String  | http://g1.globo.com/mundo.... | - |
| mainSection     | String  | Mundo | Url section (usually is a  |
| title           | String  | Rússia diz que Finlândia e Suécia... | - |
| titleCharCount  | Integer | 232 | Char count |
| textCharCount   | Integer | 493 | Char count (word count would also be possible) |
| body            | String | Non-Logged | Plain text |
| title           | String | String | Article's title |
| topics          | String | Array<String> | WIP: list of the main topics covered by the article |

## Next Steps / Questions

* Which file format would be more appropriate to generate a bigger sample (Parquet, tsv or json)?
* We are building a CNN model to represent articles as embeddings (we could replace the text column in the near future if needed)
* Please send any suggestions about the schema that would make it easier to build the metrics. 

## License

TBD / TODO
<script lang="ts">
    import * as d3 from 'd3';

    export let algoname: string;
    export let dname: string;

    function read_csv(path) {
        let request = new XMLHttpRequest();  
        request.open("GET", path, false);   
        request.send(null);

        let csvData = new Array();
        let jsonObject = request.responseText.split(/\r?\n|\r/);
        for (let i = 0; i < jsonObject.length; i++) {
            csvData.push(jsonObject[i].split(','));
        }
        csvData.splice(0, 1); // remove labels
        csvData.forEach(d => {
            d[0] = +d[0];
            d[1] = +d[1];
            d[2] = +d[2];
        })
        return csvData
    }

    let data = read_csv("results/" + dname + "/" + algoname + ".csv")

    let svgWidth = 600
    let svgHeight = 600
    const margin = { top: 25, right: 25, bottom: 25, left: 25 };

    let width = svgWidth - margin.left - margin.right
    let height = svgHeight - margin.top - margin.bottom

    let r = 4;
    let trg = 3;
    let rec = 4;
    
	const [minX, maxX] = d3.extent(data,(d) => d[0]);
    const [minY, maxY] = d3.extent(data,(d) => d[1]);

	$: xScale = d3.scaleLinear()
		.domain([minX, maxX])
        .range([0, width]);
        
	$: yScale = d3.scaleLinear()
		.domain([minY, maxY])
        .range([height, 0]);

    const colorRange = ["blue","green", "red"];

</script>

<div class="outer">
    <div class="inner">
        <svg {width} {height}>
            {#each data as d}
                {#if d[2] == 0}
                    <polygon fill="{colorRange[0]}" stroke="{colorRange[0]}" stroke-width="2" points="{xScale(d[0])-trg},{yScale(d[1])+trg} {xScale(d[0])},{yScale(d[1])-trg * 0.5} {xScale(d[0])+trg},{yScale(d[1])+trg}"></polygon>
                {:else if d[2] == 1}
                    <rect x="{xScale(d[0])}" y="{yScale(d[1])}" width="{rec}" height="{rec}" fill="{colorRange[1]}"></rect>
                {:else}
                    <circle class="circle-line"
                        r={r}
                        cx='{xScale(d[0])}'
                        cy='{yScale(d[1])}'
                        fill='{colorRange[2]}'
                    ></circle>
                {/if}
            {/each}
        </svg>
    </div>
</div>
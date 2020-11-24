<!-- https://svelte.dev/examples#scatterplot -->
<!-- https://svelte.recipes/components/scatterplot/ -->
<!-- https://svelte.dev/repl/b4c485ee69484fd8a63b8dc07c3b20a2?version=3.4.1 -->

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
        })
        return csvData
    }

    let data = read_csv("results/" + dname + "/" + algoname + ".csv")
    
    let svgWidth = 450
    let svgHeight = 450
    const margin = { top: 25, right: 25, bottom: 25, left: 25 };

    let width = svgWidth - margin.left - margin.right
    let height = svgHeight - margin.top - margin.bottom

    let r = 3;
    
	const [minX, maxX] = d3.extent(data,(d) => d[0]);
    const [minY, maxY] = d3.extent(data,(d) => d[1]);

	$: xScale = d3.scaleLinear()
		.domain([minX, maxX])
        .range([0, width]);
        
	$: yScale = d3.scaleLinear()
		.domain([minY, maxY])
        .range([height, 0]);

</script>

<div class="outer">
    <div class="inner" style='margin-right: 100px'>
        <svg {width} {height}>
            <!-- <text fill="currentColor" x="400px" y="400px">asdas</text> -->
            {#each data as d}
                <circle class="circle-line"
                    r={r}
                    cx='{xScale(d[0])}'
                    cy='{yScale(d[1])}'
                    fill='{d[2]}'
                ></circle>
            {/each}
        </svg>
        <!-- <h2>{algoname.toUpperCase()}</h2> -->
        <h2>{algoname.replace(/_/g,' ')}</h2>
        <!-- <h2>{dname.replace(/_/g,' ').toUpperCase()}</h2> -->
    </div>
</div>
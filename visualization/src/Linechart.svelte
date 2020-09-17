<!-- https://svelte.dev/repl/area-chart?version=3.24.1 -->
<!-- https://github.com/sveltejs/svelte/issues/3050 -->
<!-- https://svelte.dev/repl/ac35bd02ee76441592b1ded00ac3c515?version=3.5.2 -->
<!-- https://svelte.dev/repl/b4c485ee69484fd8a63b8dc07c3b20a2?version=3.4.1 -->
<!-- https://svelte.dev/repl/da70a84eb31c4ddda94122ae17768c19?version=3.17.2 -->

<script>
	import * as d3 from 'd3';
    import data from './stability.json'

    function translate(x, y) {
        return 'translate(' + x + ',' + y + ')'
    }

    let measures = ["umato", "umap", "atsne"]

    let r = 5;
	
    const svgWidth = 650;
    const svgHeight = 650;
    const margin = { top: 20, right: 15, bottom: 30, left: 25 };
    let width = svgWidth - margin.left - margin.right
    let height = svgHeight - margin.top - margin.bottom
    
	data.forEach((d) => {
        d.percentage = +d.percentage;
        d.tsne = +d.tsne;
		d.umap = +d.umap;
		d.atsne = +d.atsne;
		d.umato = +d.umato;
    });

    const yTicks = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10];
    const xTicks = [0,10,20,30,40,50,60,70,80,90,100];
    
	$: xScale = d3.scaleLinear()
		.domain([minX, maxX])
		.range([margin.left, width - margin.right]);

	$: yScale = d3.scaleLinear()
		.domain([Math.min.apply(null, yTicks), Math.max.apply(null, yTicks)])
        .range([height - margin.bottom, margin.top]);

	$: minX = d3.min(data, d => d.percentage);
    $: maxX = d3.max(data, d => d.percentage);
    
    let path;
    let path2;
    let path3;

    $: {
        path = `M${data.map(d => `${xScale(d.percentage)}, ${yScale(d.umato)}`).join('L')}`;
        path2 = `M${data.map(d => `${xScale(d.percentage)}, ${yScale(d.umap)}`).join('L')}`;
        path3 = `M${data.map(d => `${xScale(d.percentage)}, ${yScale(d.atsne)}`).join('L')}`;
    }
    // $: dot = data.map(d => `${xScale(d.percentage)}, ${yScale(d.dtm01)}`);
    // $: area = `${path}L${xScale(maxX)},${yScale(0)}L${xScale(minX)},${yScale(0)}Z`;

	function formatMobile(tick) {
		return "'" + tick % 100;
    }
    
    function hoverState(e, d, i) {
        hs = true;
        targetIndex = i;
    }
    
    function hoverStateOut(e, d, i) {
        hs = false;
        targetIndex = -1;
    }

</script>

<svg width={width} height={height}>
    <!-- y axis -->
    <g class="axis y-axis" transform="translate(0, {margin.top})">
        {#each yTicks as tick}
            <g class="tick tick-{tick}" transform="translate(0, {yScale(tick) - margin.bottom})">
                <line x2="100%"></line>
                <text x="-20" y="-4">{tick}</text>
                <!-- <text y="-4">{tick} {tick === 1 ? '.0 y axis' : ''}</text> -->
            </g>
        {/each}
    </g>

    <!-- x axis -->
    <g class="axis x-axis">
        {#each xTicks as tick}
            <g class="tick tick-{ tick }" transform="translate({xScale(tick)},{height})">
                <line y1="-{height}" y2="-{margin.bottom}" x1="0" x2="0"></line>
                <text y="-{margin.bottom-5}">{width > 380 ? tick : formatMobile(tick)}</text>
            </g>
        {/each}
    </g>

    <!-- x axis label -->
    <text x='{(width - margin.left) / 2 - 10}' y='{height}' font-size="12px">Percentage (%)</text>


    <!-- line chart -->
    <path class="path-line path-line-umato" d={path}></path>
    <path class="path-line path-line-umap" d={path2}></path>
    <path class="path-line path-line-atsne" d={path3}></path>
    <!-- <path class="path-area" d={area}></path> -->

    <!-- circles -->
    {#each data as d, i}
        <circle class="circle-line circle-umato"
            r={r}
            cx='{xScale(d.percentage)}'
            cy='{yScale(d.umato)}'
        ></circle>
        <circle class="circle-line circle-umap"
            r={r}
            cx='{xScale(d.percentage)}'
            cy='{yScale(d.umap)}'
        ></circle>
        <circle class="circle-line circle-atsne"
            r={r}
            cx='{xScale(d.percentage)}'
            cy='{yScale(d.atsne)}'
        ></circle>
    {/each}

    <!-- legend -->
    {#each measures as measure, i}
        <circle class="circle-line circle-{measure}"
            r={r}
            cx='{width - 90}'
            cy='{17 * (i + 1)}'
        ></circle>
        <text x='{width - 70}' y='{19 * (i+1)}' font-size="12px">{measure}</text>
    {/each}

</svg>

<style>

</style>
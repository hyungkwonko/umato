<!-- https://svelte.dev/repl/3dcdfc8c7aab4ca5a82f912f377a25e2?version=3.17.3 -->

<script lang="ts">
    import * as d3 from 'd3';

    function translate(x, y) {
        return 'translate(' + x + ',' + y + ')'
    }

    let width = 80
    let height = 450
    let arrowlength = 40

</script>

<div class="outer">
    <div class="inner">
        <svg {width} {height}>
            <!-- defining arrow head -->
            <defs>
                <marker id="arrowhead" viewBox="-10 -10 20 20" markerWidth="17" markerHeight="17" orient="auto">
                    <path d="M-6,-6 L 0,0 L -6,6" fill="black"/>
                </marker>
            </defs>
            <!-- defining arrow path -->
            <line x1={(width - arrowlength) / 2}
                x2={(width + arrowlength) / 2}
                y1={height / 2} y2={height / 2}
                stroke='black'
                stroke-width=2
                marker-end='url(#arrowhead)'></line>
        </svg>

    </div>
</div>
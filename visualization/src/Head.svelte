<!-- https://svelte.dev/repl/3dcdfc8c7aab4ca5a82f912f377a25e2?version=3.17.3 -->

<script lang="ts">
    import * as d3 from 'd3';

    function translate(x, y) {
        return 'translate(' + x + ',' + y + ')'
    }

    export let dname: string;

    let width = 80
    let height = 450
    let fontsize = 40

</script>

<div class="outer">
    <div class="inner">
        <svg {width} {height}>
            <text transform='{translate(width / 2, height / 2 + dname.length * 5)} rotate(270)' font-size='{fontsize}px'>
                {dname.charAt(0).toUpperCase() + dname.slice(1)}
            </text>
        </svg>
    </div>
</div>
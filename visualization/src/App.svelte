<script lang="ts">
	import Scatterplot from "./Scatterplot.svelte";
	import Scatterplot2 from "./Scatterplot2.svelte";
	import Arrow from "./Arrow.svelte";
	let dnames = ["spheres", "mnist", "fmnist", "kmnist"];
	let comparison = false;

</script>

<main>
	{#if comparison}
		<h1>{dnames[0].charAt(0).toUpperCase() + dnames[0].slice(1)} data</h1>
		<table>
			<tr>
				<td><Scatterplot dname={dnames[0]} algoname={"pca"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[0]} algoname={"tsne"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[0]} algoname={"umap"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[0]} algoname={"topoae"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[0]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>

		<h1>{dnames[1].charAt(0).toUpperCase() + dnames[1].slice(1)} data</h1>
		<table>
			<tr>
				<td><Scatterplot dname={dnames[1]} algoname={"pca"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"tsne"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"umap"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"topoae"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>

		<h1>{dnames[2].charAt(0).toUpperCase() + dnames[2].slice(1)} data</h1>
		<table>
			<tr>
				<td><Scatterplot dname={dnames[2]} algoname={"pca"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"tsne"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"umap"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"topoae"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>

		<h1>{dnames[3].charAt(0).toUpperCase() + dnames[3].slice(1)} data</h1>
		<table>
			<tr>
				<td><Scatterplot dname={dnames[3]} algoname={"pca"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"tsne"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"umap"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"topoae"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>
	{:else}
		<h1>UMAP pipeline</h1>
		<table>
			<tr>
				<td><Scatterplot2 dname={dnames[0]} algoname={"pca"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 dname={dnames[0]} algoname={"umap"}></Scatterplot2></td>
			</tr>
		</table>
		<h1>UMATO pipeline</h1>
		<table>
			<tr>
				<td><Scatterplot2 dname={dnames[0]} algoname={"pca"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 dname={dnames[0]} algoname={"tsne"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 dname={dnames[0]} algoname={"umap"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 dname={dnames[0]} algoname={"umap"}></Scatterplot2></td>
			</tr>
		</table>
	{/if}
</main>

<!-- svg.append("line")
  .attr("x1", 50)
  .attr("y1", 50)
  .attr("x2", 100)
  .attr("y2", 100)
  .attr("stroke", "#000")
  .attr("stroke-width", 5)
  .attr("marker-end", "url(#arrow)"); -->
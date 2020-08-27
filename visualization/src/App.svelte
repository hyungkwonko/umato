<script lang="ts">
	import Scatterplot from "./Scatterplot.svelte";
	let dnames = ["spheres", "mnist", "fmnist", "kmnist"];
</script>

<main>
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
</main>
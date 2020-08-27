<script lang="ts">
	import Scatterplot from "./Scatterplot.svelte";
	import Scatterplot2 from "./Scatterplot2.svelte";
	import Arrow from "./Arrow.svelte";
	let dnames = ["spheres", "mnist", "fmnist", "kmnist"];
	let comparison = false;

</script>

<main>
	{#if comparison}
		{#each dnames as dname}
			<h1>{dname.charAt(0).toUpperCase() + dname.slice(1)} data</h1>
			<table>
				<tr>
					<td><Scatterplot dname={dname} algoname={"pca"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"tsne"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umap"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"topoae"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umato"}></Scatterplot></td>
				</tr>
			</table>
		{/each}
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
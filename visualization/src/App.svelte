<script lang="ts">
	import Scatterplot from "./Scatterplot.svelte";
	import Scatterplot2 from "./Scatterplot2.svelte";
	import Scatterplot3 from "./Scatterplot3.svelte";
	import Linechart from "./Linechart.svelte";
	import Arrow from "./Arrow.svelte";
	import Head from "./Head.svelte";
	let dnames = ["spheres", "mnist", "fmnist", "kmnist"];
	let type = 5;

</script>

<main>
	{#if type == 0}
		<table>
			<tr>
				<td><Scatterplot3 dname={"classification"} algoname={"umato"}></Scatterplot3></td>
			</tr>
		</table>
	{:else if type == 1}
		<table>
			<tr>
				<td><Scatterplot dname={"multiphase"} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={"multiphase"} algoname={"umato_3-phases"}></Scatterplot></td>
				<td><Scatterplot dname={"multiphase"} algoname={"umato_4-phases"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 2}
		{#each dnames as dname}
			<!-- <h1>{dname.charAt(0).toUpperCase() + dname.slice(1)} data</h1> -->
			<table>
				<tr>
					<td><Head dname={dname}></Head></td>
					<td><Scatterplot dname={dname} algoname={"pca"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"tsne"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umap"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"topoae"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"atsne"}></Scatterplot></td>
					<td><Scatterplot dname={dname} algoname={"umato"}></Scatterplot></td>
				</tr>
			</table>
		{/each}
	{:else if type == 3}
		<table>
			<tr>
				<td><Linechart></Linechart></td>
			</tr>
		</table>
	{:else if type == 4}
		<table>
			<tr>
				<td><Scatterplot dname={"local"} algoname={"spheres_lr1.0"}></Scatterplot></td>
				<td><Scatterplot dname={"local"} algoname={"mnist_lr1.0"}></Scatterplot></td>
				<td><Scatterplot dname={"local"} algoname={"fmnist_lr1.0"}></Scatterplot></td>
				<td><Scatterplot dname={"local"} algoname={"kmnist_lr1.0"}></Scatterplot></td>
			</tr>
		</table>
	{:else if type == 5}
		<table>
			<tr>
				<td><Scatterplot dname={dnames[0]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[1]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[2]} algoname={"umato"}></Scatterplot></td>
				<td><Scatterplot dname={dnames[3]} algoname={"umato"}></Scatterplot></td>
			</tr>
		</table>
	{:else}
		<!-- <h1>UMATO pipeline</h1> -->
		<table>
			<tr>
				<td><Scatterplot2 filename={"(A)_global_init"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(B)_global_opt"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(C)_local_init"}></Scatterplot2></td>
				<td><Arrow></Arrow></td>
				<td><Scatterplot2 filename={"(D)_final_result"}></Scatterplot2></td>
			</tr>
		</table>
	{/if}
</main>
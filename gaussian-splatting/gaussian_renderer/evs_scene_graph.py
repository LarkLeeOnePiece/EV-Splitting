"""
EVS Scene-Graph: Memory-efficient EVS splitting implementation.

This module provides a scene-graph based approach to EVS (Event-based Gaussian Splitting)
that reduces memory consumption by:
1. Using references to original data instead of cloning
2. Storing only incremental data (newly generated child Gaussians)
3. Lazy assembly - only allocating final tensors when needed for rendering

Usage:
    # Create scene graph (no memory allocation for original data)
    graph = EVSSceneGraph.create(means3D, scales, rotations, opacity, shs)

    # Perform splits (only stores incremental data)
    splitter = EVSSplitter(evs_rasterizer)
    graph, num_splits = splitter.split_generation(graph, generation_idx=0, ...)

    # Assemble for rendering (single allocation)
    final_data = EVSSceneAssembler.assemble(graph)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitGeneration:
    """
    Represents one generation of split Gaussians.

    Each split produces 2 children from 1 parent, so if S parents are split,
    this generation contains 2*S children.

    Attributes:
        parent_indices: [S] indices of parents that were split
        means3D: [2S, 3] positions of child Gaussians
        scales: [2S, 3] scales of child Gaussians
        rotations: [2S, 4] quaternions of child Gaussians
        opacity: [2S, 1] opacities of child Gaussians
        sh_source_indices: [2S] indices into original SH for each child
        generation: generation number (0-indexed)
    """
    parent_indices: torch.Tensor    # [S] - which parent was split
    means3D: torch.Tensor           # [2S, 3]
    scales: torch.Tensor            # [2S, 3]
    rotations: torch.Tensor         # [2S, 4]
    opacity: torch.Tensor           # [2S, 1]
    sh_source_indices: torch.Tensor # [2S] - index into original SH
    generation: int                 # 0-indexed generation number
    keep_mask: torch.Tensor = None  # [2S] bool - True if NOT split in subsequent generation

    def __post_init__(self):
        """Initialize keep_mask if not provided"""
        if self.keep_mask is None:
            # Default: all children are kept (not split yet)
            self.keep_mask = torch.ones(self.means3D.shape[0], dtype=torch.bool, device=self.means3D.device)

    @property
    def num_splits(self) -> int:
        """Number of parents that were split"""
        return self.parent_indices.shape[0]

    @property
    def num_children(self) -> int:
        """Number of child Gaussians that are KEPT (not split in subsequent generation)"""
        return self.keep_mask.sum().item()

    @property
    def total_children(self) -> int:
        """Total number of child Gaussians (including those split later)"""
        return self.means3D.shape[0]

    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes"""
        total = 0
        total += self.parent_indices.numel() * self.parent_indices.element_size()
        total += self.means3D.numel() * self.means3D.element_size()
        total += self.scales.numel() * self.scales.element_size()
        total += self.rotations.numel() * self.rotations.element_size()
        total += self.opacity.numel() * self.opacity.element_size()
        total += self.sh_source_indices.numel() * self.sh_source_indices.element_size()
        total += self.keep_mask.numel() * 1  # bool = 1 byte
        return total


@dataclass
class EVSSceneGraph:
    """
    Scene graph for tracking EVS splits without copying original data.

    Memory Strategy:
    - original_* fields are REFERENCES to the input tensors (no copy)
    - keep_mask tracks which originals are NOT split
    - generations stores only the incremental child data

    Attributes:
        original_means3D: [P, 3] reference to original positions
        original_scales: [P, 3] reference to original scales
        original_rotations: [P, 4] reference to original rotations
        original_opacity: [P, 1] reference to original opacities
        original_shs: [P, K, 3] reference to original SH coefficients
        keep_mask: [P] bool - True if original Gaussian is NOT split
        generations: list of SplitGeneration for each pass
        total_split_count: total number of splits across all generations
    """
    # References to original data (NO COPY)
    original_means3D: torch.Tensor    # [P, 3]
    original_scales: torch.Tensor     # [P, 3]
    original_rotations: torch.Tensor  # [P, 4]
    original_opacity: torch.Tensor    # [P, 1]
    original_shs: torch.Tensor        # [P, K, 3]

    # Tracking which originals are kept
    keep_mask: torch.Tensor           # [P] bool - True if NOT split

    # Split generations (incremental data only)
    generations: List[SplitGeneration] = field(default_factory=list)

    # Statistics
    total_split_count: int = 0

    @classmethod
    def create(
        cls,
        means3D: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        shs: torch.Tensor
    ) -> 'EVSSceneGraph':
        """
        Create a scene graph with references to original data.

        IMPORTANT: This does NOT copy the input tensors. The caller must ensure
        the input tensors remain valid for the lifetime of this scene graph.
        """
        P = means3D.shape[0]
        device = means3D.device

        return cls(
            original_means3D=means3D,      # Reference, not copy
            original_scales=scales,         # Reference, not copy
            original_rotations=rotations,   # Reference, not copy
            original_opacity=opacity,       # Reference, not copy
            original_shs=shs,               # Reference, not copy
            keep_mask=torch.ones(P, dtype=torch.bool, device=device),
            generations=[],
            total_split_count=0
        )

    @property
    def original_count(self) -> int:
        """Number of original Gaussians"""
        return self.original_means3D.shape[0]

    @property
    def kept_count(self) -> int:
        """Number of original Gaussians that are kept (not split)"""
        return self.keep_mask.sum().item()

    @property
    def total_children(self) -> int:
        """Total number of child Gaussians across all generations"""
        return sum(g.num_children for g in self.generations)

    @property
    def final_count(self) -> int:
        """Final number of Gaussians after all splits"""
        return self.kept_count + self.total_children

    @property
    def device(self) -> torch.device:
        return self.original_means3D.device

    @property
    def dtype(self) -> torch.dtype:
        return self.original_means3D.dtype


class EVSMemoryManager:
    """
    Manages memory-efficient EVS operations.

    Provides utilities for:
    - Creating scene graphs with minimal memory overhead
    - Estimating memory usage
    - Comparing naive vs scene-graph approaches
    """

    @staticmethod
    def estimate_naive_memory(P: int, num_splits: int, num_passes: int) -> Dict[str, float]:
        """
        Estimate memory usage of naive implementation.

        Args:
            P: number of original Gaussians
            num_splits: average splits per pass
            num_passes: number of passes

        Returns:
            dict with memory estimates in MB
        """
        # Per Gaussian: means(12) + scales(12) + rotations(16) + opacity(4) + SH(~192) = ~236 bytes
        bytes_per_gaussian = 236

        # Clone at start
        clone_mb = P * bytes_per_gaussian / (1024 * 1024)

        # Each pass grows the working set
        working_sizes = [P]
        for _ in range(num_passes):
            current = working_sizes[-1]
            # Assume num_splits are split, creating 2*num_splits children
            # New size = (current - num_splits) + 2*num_splits = current + num_splits
            working_sizes.append(current + num_splits)

        # Peak is at the end
        peak_working_mb = max(working_sizes) * bytes_per_gaussian / (1024 * 1024)

        # Total peak includes clone + working set
        total_peak_mb = clone_mb + peak_working_mb

        return {
            'clone_mb': clone_mb,
            'peak_working_mb': peak_working_mb,
            'total_peak_mb': total_peak_mb
        }

    @staticmethod
    def estimate_scenegraph_memory(graph: EVSSceneGraph) -> Dict[str, float]:
        """
        Estimate current memory usage of scene-graph approach.

        Args:
            graph: the EVSSceneGraph

        Returns:
            dict with memory estimates in MB
        """
        # Keep mask
        mask_bytes = graph.keep_mask.numel() * 1  # bool = 1 byte

        # Generations
        gen_bytes = sum(g.memory_bytes() for g in graph.generations)

        # Total overhead (original data is not counted as it's shared)
        overhead_bytes = mask_bytes + gen_bytes

        return {
            'mask_mb': mask_bytes / (1024 * 1024),
            'generations_mb': gen_bytes / (1024 * 1024),
            'total_overhead_mb': overhead_bytes / (1024 * 1024)
        }


class EVSSplitter:
    """
    Performs EVS splitting with scene-graph updates.

    This class wraps the CUDA EVS rasterizer and updates the scene graph
    with minimal memory allocations.
    """

    def __init__(self, evs_rasterizer):
        """
        Args:
            evs_rasterizer: the EVSplitter instance with CUDA split() method
        """
        self.evs_rasterizer = evs_rasterizer

    def split_generation(
        self,
        graph: EVSSceneGraph,
        generation_idx: int,
        means2D: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ) -> Tuple[EVSSceneGraph, int]:
        """
        Perform one generation of splitting.

        For generation 0: split from original Gaussians
        For generation N>0: split from generation N-1's children

        Args:
            graph: the scene graph to update
            generation_idx: which generation this is (0-indexed)
            means2D: optional 2D means (usually None, computed internally)
            colors_precomp: optional precomputed colors
            cov3D_precomp: optional precomputed 3D covariance

        Returns:
            (updated_graph, num_splits)
        """
        if generation_idx == 0:
            # Split from original data
            return self._split_from_original(
                graph, means2D, colors_precomp, cov3D_precomp
            )
        else:
            # Split from previous generation's children
            return self._split_from_generation(
                graph, generation_idx, means2D, colors_precomp, cov3D_precomp
            )

    def _split_from_original(
        self,
        graph: EVSSceneGraph,
        means2D: Optional[torch.Tensor],
        colors_precomp: Optional[torch.Tensor],
        cov3D_precomp: Optional[torch.Tensor],
    ) -> Tuple[EVSSceneGraph, int]:
        """Split from original Gaussians (generation 0)"""

        # Get indices of Gaussians that haven't been split yet
        active_indices = torch.where(graph.keep_mask)[0]

        if active_indices.numel() == 0:
            return graph, 0

        # Extract active subset for split computation
        active_means = graph.original_means3D[active_indices]
        active_scales = graph.original_scales[active_indices]
        active_rotations = graph.original_rotations[active_indices]
        active_opacity = graph.original_opacity[active_indices]
        active_shs = graph.original_shs[active_indices]

        # Call CUDA EVS splitter
        num_splits, split_flag, split_offsets, means_new, scales_new, \
            rots_new, opacity_new, src_ids = self.evs_rasterizer.split(
                means3D=active_means,
                means2D=means2D,
                shs=active_shs,
                colors_precomp=colors_precomp,
                opacities=active_opacity,
                scales=active_scales,
                rotations=active_rotations,
                cov3D_precomp=cov3D_precomp
            )

        torch.cuda.synchronize()
        num_gs_split = torch.count_nonzero(split_flag).item()

        if num_gs_split == 0:
            return graph, 0

        # Update keep_mask: mark split Gaussians as not kept
        split_mask_local = (split_flag == 1)
        # Map local indices back to global original indices
        local_split_indices = torch.where(split_mask_local)[0]
        global_split_indices = active_indices[local_split_indices]
        graph.keep_mask[global_split_indices] = False

        # Compute SH source indices for children
        # src_ids tells us which active Gaussian was split (local index)
        # We need to map to original indices
        src_ids_long = src_ids.to(torch.int64)

        # Clamp src_ids to valid range
        P_active = active_indices.numel()
        if num_splits > 0:
            src_ids_clamped = torch.clamp(src_ids_long, 0, P_active - 1)
        else:
            src_ids_clamped = src_ids_long

        # Map to original indices: each split parent produces 2 children
        sh_source_indices = active_indices[src_ids_clamped].repeat_interleave(2)

        # Ensure opacity has correct shape
        if opacity_new.dim() == 1:
            opacity_new = opacity_new.unsqueeze(1)

        # Create new generation
        new_gen = SplitGeneration(
            parent_indices=src_ids,
            means3D=means_new,
            scales=scales_new,
            rotations=rots_new,
            opacity=opacity_new,
            sh_source_indices=sh_source_indices,
            generation=0
        )

        graph.generations.append(new_gen)
        graph.total_split_count += num_gs_split

        return graph, num_gs_split

    def _split_from_generation(
        self,
        graph: EVSSceneGraph,
        generation_idx: int,
        means2D: Optional[torch.Tensor],
        colors_precomp: Optional[torch.Tensor],
        cov3D_precomp: Optional[torch.Tensor],
    ) -> Tuple[EVSSceneGraph, int]:
        """Split from previous generation's children (generation > 0)"""

        if generation_idx <= 0 or generation_idx > len(graph.generations):
            return graph, 0

        # Get previous generation
        prev_gen = graph.generations[generation_idx - 1]

        # Source data is the children from previous generation
        source_means = prev_gen.means3D
        source_scales = prev_gen.scales
        source_rotations = prev_gen.rotations
        source_opacity = prev_gen.opacity

        # For SH, use the source indices to get from original
        source_shs = graph.original_shs[prev_gen.sh_source_indices]

        # Call CUDA EVS splitter on all children
        num_splits, split_flag, split_offsets, means_new, scales_new, \
            rots_new, opacity_new, src_ids = self.evs_rasterizer.split(
                means3D=source_means,
                means2D=means2D,
                shs=source_shs,
                colors_precomp=colors_precomp,
                opacities=source_opacity,
                scales=source_scales,
                rotations=source_rotations,
                cov3D_precomp=cov3D_precomp
            )

        torch.cuda.synchronize()
        num_gs_split = torch.count_nonzero(split_flag).item()

        if num_gs_split == 0:
            return graph, num_gs_split

        # For children from this generation, trace SH back through parent
        src_ids_long = src_ids.to(torch.int64)
        P_source = source_means.shape[0]
        src_ids_clamped = torch.clamp(src_ids_long, 0, P_source - 1)

        # The SH source for new children comes from the parent's SH source
        sh_source_indices = prev_gen.sh_source_indices[src_ids_clamped].repeat_interleave(2)

        # Ensure opacity has correct shape
        if opacity_new.dim() == 1:
            opacity_new = opacity_new.unsqueeze(1)

        # Update previous generation's keep_mask: mark split children as not kept
        split_mask = (split_flag == 1)
        prev_gen.keep_mask[split_mask] = False

        # Create new generation
        new_gen = SplitGeneration(
            parent_indices=src_ids,
            means3D=means_new,
            scales=scales_new,
            rotations=rots_new,
            opacity=opacity_new,
            sh_source_indices=sh_source_indices,
            generation=generation_idx
        )

        graph.generations.append(new_gen)
        graph.total_split_count += num_gs_split

        return graph, num_gs_split


class EVSSceneAssembler:
    """
    Lazily assembles the final scene for rendering.

    This is called once before actual rasterization, performing a single
    allocation for the final tensors.
    """

    @staticmethod
    def assemble(
        graph: EVSSceneGraph,
        evs_debug: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble final Gaussian tensors for rendering.

        Memory strategy:
        - Allocate output tensors of exact final size (single allocation)
        - Copy kept originals using index gather
        - Copy children from each generation

        Args:
            graph: the EVSSceneGraph with all split information
            evs_debug: if True, color children red/green for visualization

        Returns:
            dict with 'means3D', 'scales', 'rotations', 'opacity', 'shs'
        """
        final_count = graph.final_count
        device = graph.device
        dtype = graph.dtype

        # Get SH shape from original
        sh_shape = graph.original_shs.shape[1:]  # (K, 3)

        # Allocate final tensors (single allocation for entire process)
        means3D_final = torch.empty(final_count, 3, device=device, dtype=dtype)
        scales_final = torch.empty(final_count, 3, device=device, dtype=dtype)
        rotations_final = torch.empty(final_count, 4, device=device, dtype=dtype)
        opacity_final = torch.empty(final_count, 1, device=device, dtype=dtype)
        shs_final = torch.empty(final_count, *sh_shape, device=device, dtype=dtype)

        # Copy kept originals using index gather
        kept_indices = torch.where(graph.keep_mask)[0]
        kept_count = kept_indices.numel()

        if kept_count > 0:
            means3D_final[:kept_count] = graph.original_means3D[kept_indices]
            scales_final[:kept_count] = graph.original_scales[kept_indices]
            rotations_final[:kept_count] = graph.original_rotations[kept_indices]
            opacity_final[:kept_count] = graph.original_opacity[kept_indices]
            shs_final[:kept_count] = graph.original_shs[kept_indices]

        # Copy children from each generation (only kept children)
        offset = kept_count
        for gen in graph.generations:
            child_count = gen.num_children  # Only counts kept children
            if child_count == 0:
                continue

            # Get indices of kept children in this generation
            kept_child_indices = torch.where(gen.keep_mask)[0]
            end = offset + child_count

            means3D_final[offset:end] = gen.means3D[kept_child_indices]
            scales_final[offset:end] = gen.scales[kept_child_indices]
            rotations_final[offset:end] = gen.rotations[kept_child_indices]
            opacity_final[offset:end] = gen.opacity[kept_child_indices]

            # SH from original via indices (only for kept children)
            kept_sh_indices = gen.sh_source_indices[kept_child_indices]
            if not evs_debug:
                shs_final[offset:end] = graph.original_shs[kept_sh_indices]
            else:
                # Debug mode: color children red/green
                shs_src = graph.original_shs[kept_sh_indices]
                shs_debug = shs_src.clone()
                red = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
                green = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                shs_debug[0::2, 0, :] = red
                shs_debug[1::2, 0, :] = green
                shs_final[offset:end] = shs_debug

            offset = end

        return {
            'means3D': means3D_final,
            'scales': scales_final,
            'rotations': rotations_final,
            'opacity': opacity_final,
            'shs': shs_final,
        }

    @staticmethod
    def get_stats(graph: EVSSceneGraph) -> Dict[str, Any]:
        """
        Get statistics about the scene graph.

        Returns:
            dict with various statistics
        """
        memory_stats = EVSMemoryManager.estimate_scenegraph_memory(graph)

        return {
            'original_count': graph.original_count,
            'kept_count': graph.kept_count,
            'total_children': graph.total_children,
            'final_count': graph.final_count,
            'num_generations': len(graph.generations),
            'total_splits': graph.total_split_count,
            'memory_overhead_mb': memory_stats['total_overhead_mb'],
        }

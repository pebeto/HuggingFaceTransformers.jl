using HuggingFaceTransformers.HFHub
using HuggingFaceTransformers.HFHub:
    RepoRef, parse_repo, repo_folder_name, default_cache_dir
using Test

@testset "parse_repo" begin
    r = parse_repo("meta-llama/Llama-3.2-1B-Instruct")
    @test r isa RepoRef
    @test r.repo_id == "meta-llama/Llama-3.2-1B-Instruct"
    @test r.revision == "main"

    r2 = parse_repo("meta-llama/Llama-3.2-1B-Instruct@refs/pr/3")
    @test r2.repo_id == "meta-llama/Llama-3.2-1B-Instruct"
    @test r2.revision == "refs/pr/3"

    r3 = parse_repo("BAAI/bge-small-en-v1.5"; default_revision="v2")
    @test r3.revision == "v2"

    @test_throws ArgumentError parse_repo("no-slash")
    @test_throws ArgumentError parse_repo("org/name@")
    @test_throws ArgumentError parse_repo("@rev")
end

@testset "repo_folder_name" begin
    @test repo_folder_name("meta-llama/Llama-3.2-1B") == "models--meta-llama--Llama-3.2-1B"
    @test repo_folder_name("BAAI/bge-small-en-v1.5") == "models--BAAI--bge-small-en-v1.5"
    @test repo_folder_name("org/name"; repo_type="dataset") == "datasets--org--name"
end

@testset "default_cache_dir env precedence" begin
    saved = Dict(
        k => get(ENV, k, nothing) for k in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME")
    )
    try
        delete!(ENV, "HF_HUB_CACHE")
        delete!(ENV, "HF_HOME")
        delete!(ENV, "XDG_CACHE_HOME")
        @test default_cache_dir() == joinpath(homedir(), ".cache", "huggingface", "hub")

        ENV["XDG_CACHE_HOME"] = "/tmp/xdg"
        @test default_cache_dir() == joinpath("/tmp/xdg", "huggingface", "hub")

        ENV["HF_HOME"] = "/tmp/hfhome"
        @test default_cache_dir() == joinpath("/tmp/hfhome", "hub")

        ENV["HF_HUB_CACHE"] = "/tmp/hub-explicit"
        @test default_cache_dir() == "/tmp/hub-explicit"
    finally
        for (k, v) in saved
            if isnothing(v)
                delete!(ENV, k)
            else
                ENV[k] = v
            end
        end
    end
end

# Network-gated integration test. Set HFT_TEST_NETWORK=1 to enable.
# Uses a HF maintainer-controlled tiny fixture repo so the cost is minimal.
if get(ENV, "HFT_TEST_NETWORK", "0") == "1"
    @testset "download_file + Python cache layout (network)" begin
        repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        mktempdir() do cache
            path = HFHub.download_file(repo_id, "config.json"; cache_dir=cache)
            @test isfile(path)
            @test occursin(
                joinpath("models--$(replace(repo_id, '/' => "--"))", "snapshots"), path
            )

            # The path should be a (relative) symlink into ../../blobs/<etag>.
            @test islink(path)
            link_target = readlink(path)
            @test startswith(link_target, joinpath("..", "..", "blobs"))
            @test isfile(realpath(path))

            # refs/<revision> should hold a commit SHA matching the snapshot dir.
            commit = basename(dirname(path))
            refs_file = joinpath(
                cache, "models--$(replace(repo_id, '/' => "--"))", "refs", "main"
            )
            @test isfile(refs_file)
            @test strip(read(refs_file, String)) == commit

            # Second call must reuse the cached blob (no re-download).
            blob_dir = joinpath(cache, "models--$(replace(repo_id, '/' => "--"))", "blobs")
            blob_before = sort(readdir(blob_dir))
            path2 = HFHub.download_file(repo_id, "config.json"; cache_dir=cache)
            @test path2 == path
            @test sort(readdir(blob_dir)) == blob_before

            # local_files_only must work after the cache is warm.
            path3 = HFHub.download_file(
                repo_id, "config.json"; cache_dir=cache, local_files_only=true
            )
            @test path3 == path
        end
    end

    @testset "snapshot_download (network)" begin
        repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        mktempdir() do cache
            snap = HFHub.snapshot_download(repo_id; cache_dir=cache)
            @test isdir(snap)
            @test isfile(joinpath(snap, "config.json"))
            # Either sharded index or a single safetensors must be present.
            @test isfile(joinpath(snap, "model.safetensors")) ||
                isfile(joinpath(snap, "model.safetensors.index.json"))
        end
    end
end

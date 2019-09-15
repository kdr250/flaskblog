$(function() {

  function buildHTML(post){
    var html = `<div class="post-content" id="post-content" data-post-id=${post.id} >
    <h1>${post.title}</h1>
    <p>投稿日: ${post.date_posted} By ${post.authorname}</p>
    <p>${post.content}</p>
    </div>`
    return html
  }

  $('#submit').on('click', function(e){
    e.preventDefault();
    console.log('hoge');
    $.ajax({
      type: 'POST',
      url: '/post_ajax',
      data: $(this).parent().parent('form').serialize(),
    })
    .done(function (data) {
      console.log('Ajax成功!')
      var html = buildHTML(data)
      $('#message-box').append(html)
      $('#post-form').get(0).reset();
      $('html, body').animate({scrollTop:$('#message-box')[0].scrollHeight});
    })
    .fail(function() {
      alert('Ajaxに失敗しました');
    })
  })
  var reloadMessages = function() {
    // lastメソッドはidではなくクラス指定(class="~" => $('.~:last'))出ないと効かない!
    // last_post_id = $('#post-content').last().data('post-id')
    var last_post_id = $('.post-content:last').data('post-id')
    console.log(last_post_id)
    $.ajax({
      //ルーティングで設定した通り/groups/id番号/api/messagesとなるよう文字列を書く
      url: '/post_api',
      //ルーティングで設定した通りhttpメソッドをgetに指定
      type: 'POST',
      // dataType: 'json',
      // //dataオプションでリクエストに値を含める
      // data: {id: last_post_id}
      data: JSON.stringify({"id":last_post_id}),
      contentType: 'application/json',
    })
    .done(function(posts) {
      //追加するHTMLの入れ物を作る
      var insertHTML = '';

      //メッセージが入ったHTMLを取得
      var messege_box = $('#message-box')

      //配列messagesの中身一つ一つを取り出し、HTMLに変換したものを入れ物に足し合わせる
      console.log(posts)

      if ($.isEmptyObject(posts) == false) {
        posts.forEach(function(post){
          var html = buildHTML(post)
          $(messege_box).append(html);
        })
        $('html, body').animate({scrollTop:$('#message-box')[0].scrollHeight});
        console.log('animated')
      } 
    })
    .fail(function() {
      alert('error');
    });
  };
  setInterval(reloadMessages, 10000);
})